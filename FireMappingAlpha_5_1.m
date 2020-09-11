% 1. Added new Laplacian generation for "N" agents. "formation" tells us
%    the type of network. 0 -> strong, 1 -> ring, 2 -> random. If random is
%    chosen, the "rand_comm" parameter must be set. 
%    0 < rand_comm < num_of_agents.
% 2. "alpha" was added from our paper.
% 3. Put laplacian generation in a function
% 4. Put displaying of true state and estimated state in a function

import java.util.LinkedList;
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ENV_SIZE = 20;                      % Size of environment

formation = 1;                      % 0 -> strong, 1 -> ring, 2 -> random
rand_comm = 3;                      % Num of random UAVs to connect with
depth = 3;                          % Depth of our tree
branch_factor = 4;                  % Number of childs per node (N,S,E,W)
num_of_agents = 5;                  % Number of UAVs simulated
fire_rate = 0.007;                  % Fire spread rate (0 <= rate <= 1)
burn_out_rate = 0;                  % Fire burnout rate (0 <= rate <= 1)
duration = 1000;                    % Duration of simulation
start_display = 0;               % Display maps after "n" timesteps
fusion_interval = 10;               % Fuse est maps after "n" timesteps
cost_power = 2;                     % Used to tune cost of proximity of UAV 
                                    % to others
UAV_range = 5;                      % Range of UAV sight (inclusive)
gamma = 3/10;                       % 0 < gamma < 1/max(num_connected)

alpha = 0.7;                        % Controls weight of reward function

false_pos = 0.10;                   % UAV Measurement inaccuracy rates
false_neg = 0.10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% True Positive and False Positive.
true_pos = 1 - false_pos;
true_neg = 1 - false_neg;

% Laplacian matrix for communication between agents. (Ring formation)
L = get_laplacian(num_of_agents,formation,rand_comm);

% Estimated matrix initialized to 50% probability of fire in each location.
% (Centralized Map)
est_state = 0.5*ones(ENV_SIZE);

% Estimated matrix for each UAV within the environment. 
est_state_UAVs = 0.5*ones(ENV_SIZE,ENV_SIZE,num_of_agents);

% Initialized ENV_SIZExENV_SIZE matrix as our true state environment 
% (binary representation).
true_state = zeros(ENV_SIZE);

% Used to keep track of coordinates of "burned out" squares.
fire_out_x = LinkedList();
fire_out_y = LinkedList();

% Generate and place random location for the initial fire.
x = randi(ENV_SIZE);    % vertical
y = randi(ENV_SIZE);    % horizontal
true_state(x,y) = 1;
fire_out_x.add(x);
fire_out_y.add(y);

% Coordinates of randomly places UAVs. 
robX = randperm(ENV_SIZE,num_of_agents);
robY = randperm(ENV_SIZE,num_of_agents);

% Pre-allocate to calculate error in our model.
total_err = zeros(duration,1);

total_err_UAVs = zeros(duration,1);
UAV_error = zeros(num_of_agents,1);

% Used to calculate "best path". Calculated beforehand to optimize 
% runtime.
paths = zeros(depth,2,branch_factor^depth);
second_value = get_directions(paths,branch_factor^depth,1,0);

% Keep spreading fire and running algorithm until time duration is up.
for index = 1:duration
    % Spread the fire throughout the environment based on the fire_rate.
    [true_state,fire_out_x,fire_out_y] = spread_fire(true_state,ENV_SIZE,fire_out_x,fire_out_y,fire_rate);
    
    % The fire spread is updated. Now perform step 2 and update
    % est_state by using prediction model. (Centralized Map)
    est_state = update_est_state(est_state,ENV_SIZE,fire_rate);
    
    % Update the estimated mapping for each UAV.
    for i = 1:size(est_state_UAVs,3)
        est_state_UAVs(:,:,i) = update_est_state(est_state_UAVs(:,:,i),ENV_SIZE,fire_rate);
    end

    for loc = 1:size(robX,2)
        % Sensor "error" chance of having incorrect information (STEP 3)
        M = true_state(robX(loc),robY(loc));
        if M == 1
            % The true state is 1 and we have a false negative
            if rand() < false_neg
                M = 0;
            end
        else
            % The true state is 0 and we have a false positive
            if rand() < false_pos
                M = 1;
            end
        end
        
        % Now that we updated est_state, we want to update based on our
        % "sensor readings" or measurements. (Step 3 & 4) (Centralized Map)
        est_state = update_est_state_2(est_state,M,robX(loc),robY(loc), ...
                                       false_pos,false_neg,true_pos,true_neg);
        
        % Update estimated state for each UAV.
        est_state_UAVs(:,:,loc) = update_est_state_2(est_state_UAVs(:,:,loc),M, ...
                                              robX(loc),robY(loc),false_pos,false_neg,true_pos,true_neg);  

        % Update connected UAV's mapping based on current measurement.
        for j = 1:size(L(:,loc),1)
            if L(j,loc) == -1
                % j is UAV connected to loc
                est_state_UAVs(:,:,j) = update_est_state_2(est_state_UAVs(:,:,j),M, ...
                                        robX(loc),robY(loc),false_pos,false_neg,true_pos,true_neg);
            end
        end
    end
    
    % Only display maps after reaching timestep "start_display". 
    if (index >= start_display)
        display(num_of_agents,ENV_SIZE,true_state,est_state,est_state_UAVs,robY,robX);
        pause(0.02);
    end
    
    % Perform estimated mapping fusion for connected agents after specified
    % simulation iterations
    if (~mod(index,fusion_interval))
        for agent = 1:num_of_agents
            est_state_UAVs(:,:,agent) = est_map_fusion(L(agent,:),est_state_UAVs,...
                                        est_state_UAVs(:,:,agent),gamma);
        end
    end
    
    % Move UAVs to the most optimal location based on maximum reward.
    for loc = 1:size(robX,2)
        % Find future paths of UAVs on Laplacian and within localized halo.
        % CURRENTLY ONLY WORKS WITH LAPLACIAN UAVS!!
        
        
        % Find the best path for each UAV based on their estimated mapping.
        best_path = find_best_path(robX(loc),robY(loc),est_state_UAVs(:,:,loc), ...
                                   depth,branch_factor,ENV_SIZE,second_value,num_of_agents, ...
                                   robX,robY,loc,cost_power,UAV_range,alpha);
                               
        % Now choose a direction to move robot. (STEP 6) (ROHAN - CHECK THIS)
        %agent_loc = best_path(1,:);
        robX(loc) = best_path(1,1);
        robY(loc) = best_path(1,2);
    end
    
    % Implementation of fire "burning out"
    if fire_out_x.size() > 0 && rand() <= burn_out_rate
        % Fire burned out. Square no longer eligible to contain fire
        i2 = fire_out_x.removeFirst();
        j2 = fire_out_y.removeFirst();
        true_state(i2,j2) = -1;
    end
    
    % Calculate Squared Error for centralized mapping.
    true_state_2 = true_state;
    true_state_2(true_state_2 == -1) = 0;
    error = true_state_2 - est_state;
    squared_err = error.^2;
    total_err(index) = sum(sum(squared_err));
    
    % Calculate Squared Error for each UAV's mapping.
    for i = 1:num_of_agents
        error_UAVs = true_state_2 - est_state_UAVs(:,:,i);
        squared_err_UAV = error_UAVs.^2;
        UAV_error(i) = sum(sum(squared_err_UAV));
    end
    total_err_UAVs(index) = sum(UAV_error) / num_of_agents;
end

fprintf('Simulation Complete! :)\n');
total_err = total_err/duration;
total_err_UAV = total_err_UAVs/duration;
toc

% Returns the Laplacian matrix based on the number of agents, formation, K.
% 0 --> strongly connected
% 1 --> ring formation
% 2 --> random communication
function L = get_laplacian(num_of_agents,formation,K)
    L = zeros(num_of_agents);
    % Ring formation
    if (formation == 1)
        for i = 2:num_of_agents
            L(i-1,i) = -1;
            L(i,i-1) = -1;
        end
        L(1,num_of_agents) = -1;
        L(num_of_agents,1) = -1;
    % random communication (K agents) 
    elseif (formation == 2)
        for i = 1:num_of_agents
            range = setdiff(1:num_of_agents, i);
            for j = 1:K
                random = range(randi(length(range)));
                while L(i,random) == -1
                    random = range(randi(length(range)));
                end
                L(i,random) = -1;
            end
        end
    % strongly connected
    else
        L = -ones(num_of_agents);
    end

    % Initialize Laplacian matrix across diagonals (Out-degree)
    for i = 1:num_of_agents
        L(i,i) = 0;
        L(i,i) = -sum(L(i,:));
    end
end

% Updates the model based on the fire spread rate and propagates the fire.
function [true_state,fire_out_x,fire_out_y] = spread_fire(true_state,ENV_SIZE,fire_out_x,fire_out_y,fire_rate)
    for i = 1:ENV_SIZE
        for j = 1:ENV_SIZE
            if true_state(i,j) == 1
                % 10% chance of fire spreading to up, left, down, and right 
                % square
                spread_up = rand();
                spread_left = rand();
                spread_down = rand();
                spread_right = rand();

                % Change true state based on if fire spread or not
                if spread_up <= fire_rate && i > 1 && true_state(i-1,j) ~= -1
                    true_state(i-1,j) = 1;
                    fire_out_x.add(i-1);
                    fire_out_y.add(j);
                end
                if spread_left <= fire_rate && j > 1 && true_state(i,j-1) ~= -1
                    true_state(i,j-1) = 1;
                    fire_out_x.add(i);
                    fire_out_y.add(j-1);
                end
                if spread_down <= fire_rate && i < ENV_SIZE && true_state(i+1,j) ~= -1
                    true_state(i+1,j) = 1;
                    fire_out_x.add(i+1);
                    fire_out_y.add(j);
                end
                if spread_right <= fire_rate && j < ENV_SIZE && true_state(i,j+1) ~= -1
                    true_state(i,j+1) = 1;
                    fire_out_x.add(i);
                    fire_out_y.add(j+1);
                end
            end
        end
    end
end

% Returns a colormap used to display the environment with "dark green"
% indicating low probability and "red" indicating high probability.
function colormap = get_colormap(L) 
    map2 = zeros(L,3); 
    
    for i = 1:50 
        map2(i,:) = [0 (i+50)/100 0];  
    end
    for i = 1:100
        map2(i+50,:) = [i/100 1 0]; 
    end
    for i = 1:100
        map2(i+150,:) = [1 (100-i)/100  0]; 
    end
    colormap = map2;
end

% Used to get the directions the agent will be able to move. Note -- There
% are illegal moves still within the directions returned.
function paths = get_directions(paths,N,L,off)
    % termination condition for each branch 
    if N==1
        return 
    end
    
    % These four lines sets the next action (U,L,D,R) 
    paths(L:end,2,(1:(N/4))+off) = paths(L:end,2,(1:(N/4))+off)-1;
    paths(L:end,1,((N/4+1):(N/2))+off) = paths(L:end,1,((N/4+1):(N/2))+off)-1;
    paths(L:end,2,((N/2+1):(3*N/4))+off) = paths(L:end,2,((N/2+1):(3*N/4))+off)+1;
    paths(L:end,1,((3*N/4+1):N)+off) = paths(L:end,1,((3*N/4+1):N)+off)+1;
    
    % Recursive function calls (U,L,D,R)
    paths = get_directions(paths,N/4,L+1,off); 
    paths = get_directions(paths,N/4,L+1,off+(N/4)); 
    paths = get_directions(paths,N/4,L+1,off+(N/2)); 
    paths = get_directions(paths,N/4,L+1,off+(3*N/4));
end

% Gets the probabilities of a specified path
function probs = get_probs(paths, path, est_state, L)
    probs = zeros(1,size(path,1));
    
    % Check all paths and rows of each path
    for row = 1:size(path,1)  
        if (paths(row,1,L) > 0 && paths(row,2,L) > 0)
            probs(row) = est_state(paths(row,1,L),paths(row,2,L));
        end
    end
end

% Gets the reward for a specified path based on its probabilities.
% Returns an integer as the reward based on the binary entropy function.
function reward = get_reward(probs)
    entropy = zeros(1,size(probs,2));
    for i = 1:size(probs,2)
        entropy(i) = -probs(i) * log2(probs(i)) - (1 - probs(i)) * log2(1 - probs(i));
    end
    reward = sum(entropy);
end

% Use prediction model to update estimated state map. (STEP 2)
function new_est_state = update_est_state(est_state,ENV_SIZE,fire_rate)
    for i = 1:size(est_state)
        for j = 1:size(est_state)
            neighbors = expand_helper(i,j,ENV_SIZE);
            neighbor_probs = zeros(size(neighbors,1),1)';%[0,0,0,0];
            for k = 1:size(neighbors,1)%4
                coord = neighbors(k,:);
                % Check for legal index!
                if coord(1) > 0 && coord(2) > 0
                    neighbor_probs(k) = est_state(coord(1),coord(2));
                end
            end
            % Now we have vector of probabilities and current probability
            curr_prob = est_state(i,j);
            
            % Calculate next state belief
            est_state(i,j) = (curr_prob + (sum(neighbor_probs)*fire_rate)) / (1 + (nnz(neighbor_probs)*fire_rate));
        end
    end
    new_est_state = est_state;
end

% Use "sensor data" to update our belief state. (STEP 3 & 4)
function new_est_state = update_est_state_2(est_state,M,i,j,false_pos,false_neg,true_pos,true_neg)
    % Sensor "error" 10% chance of having incorrect information (STEP 3)
%     if M == 1
%         % The true state is 1 and we have a false negative
%         if rand() < false_neg
%             M = 0;
%         end
%     else
%         % The true state is 0 and we have a false positive
%         if rand() < false_pos
%             M = 1;
%         end
%     end

    prior = est_state(i,j);
    prior_comp = 1 - prior;
    % Now M is our "sensor reading". Update est_state (STEP 4)
    if M == 1
        % true positive or false positive
        temp = (false_pos * prior_comp) + (true_pos * prior);
        prob = (true_pos * prior) / temp;
    else
        % false negative or true negative
        temp = (true_neg * prior_comp) + (false_neg * prior);
        prob = (false_neg * prior) / temp;
    end
    est_state(i,j) = prob;
    new_est_state = est_state;
end

% Finds the optimal path for a UAV to take based on the entropy/reward of 
% each path. Depth specifies the number of discrete steps into the future 
% to be analyzed. 
function best_path = find_best_path(i,j,est_state,depth,branch_factor,ENV_SIZE,second_value,...
                                    num_of_agents,robX,robY,index,power,range,alpha)
    first_value = zeros(depth,2,branch_factor^depth);
    first_value(:,1,:) = i;
    first_value(:,2,:) = j;

    % U U U U   U U U U   U U U U   U U U U 
    % U U U U   L L L L   D D D D   R R R R ... change U --> L and 
    % U L D R   U L D R   U L D R   U L D R     keep going (64 total)

    % One (3x2) matrix per column, illegal moves should be zero matrix
    paths = first_value + second_value;

    % Check all matrix "sheets" for indexes > ENV_SIZE or indexes < 1
    for k = 1:branch_factor^depth
        % Range of indexes
        B = paths(:,:,k);

        % Now check if matrix values are within range
        C = any(B > ENV_SIZE);

        D = any(B < 1);
        if any(C(:) > 0) || any(D(:) > 0)
            paths(:,:,k) = zeros(depth,2);
        end
    end

    % Generate proximity matrix based on UAV position up, down, left, right
    % Order is left, up, right, down
    proximity = zeros(num_of_agents,num_of_agents,4);
    proximity(:,:,1) = UAV_proximity_cost(i,j-1,num_of_agents,robX,robY,power,range);
    proximity(:,:,2) = UAV_proximity_cost(i-1,j,num_of_agents,robX,robY,power,range);
    proximity(:,:,3) = UAV_proximity_cost(i,j+1,num_of_agents,robX,robY,power,range);
    proximity(:,:,4) = UAV_proximity_cost(i+1,j,num_of_agents,robX,robY,power,range);
    
    % Now all paths are generated and illegal paths are zeroed out
    rewards = zeros(1,branch_factor^depth);
    count = 1;
    break_val = (branch_factor^depth) / 4;
    for L = 1:branch_factor^depth
        probs = get_probs(paths, paths(:,:,L), est_state, L);
        
        % Find reward using Binary Entropy Function 
        % (SUBTRACT OFF COST OF BEING CLOSE TO ANOTHER UAV)
        rewards(L) = (alpha*get_reward(probs)) - ((1-alpha)*sum(proximity(index,:,count)));
        
        % If we have reached the "next" direction, we need to evaluate
        % matrix for that specific direction. ex. up --> left
        if (mod(L,break_val) == 0)
            count = count + 1;
        end
    end

    % Now we have a 1,64 vector of rewards. The max index will give 
    % us the "best" path to take for our agent
    [~, ind] = max(rewards);

    % Most optimal path (For first step, this will always be the first
    % expanded path since we initialize our est_state with uniform 
    % probabilities.
    best_path = paths(:,:,ind);
end

% Calculates the cost for each UAV path based on proximity to other agents
% in the environment. 
function cost = UAV_proximity_cost(i,j,num_of_agents,robX,robY,power,range)
%     power = 2;          % Used to tune cost of proximity of UAV to others
%     range = 5;          % Range of UAV sight (inclusive)
    
    cost1 = zeros(num_of_agents);
    for z = 1:num_of_agents
        cost_vector = inf*ones(1,num_of_agents);
        for x = 1:num_of_agents
            distance = abs(robX(x)-i) + abs(robY(x)-j);
            if distance <= range
                cost_vector(x) = distance;
            end
        end
        cost1(z,:) = cost_vector;
    end
    
    cost = 1./(cost1.^power);
    cost(logical(eye(size(cost)))) = 0;
end

% Fuses estimated mappings of UAVs based on the communication network
% specified by the Laplacian matrix.
function fused_est_map = est_map_fusion(L,est_state_UAVs,my_est_state,gamma)
%     gamma = 3/10;                    % 0 < gamma < 1/(num_connected + 1)
    for j = 1:size(L,2)
        if (L(j) == -1)
            my_est_state = my_est_state.*(est_state_UAVs(:,:,j)./my_est_state).^gamma;  
        end
    end
    fused_est_map = my_est_state;
end

% Displays the estimated states (UAVs and centralized) and true state 
% (centralized)
function display(num_of_agents,ENV_SIZE,true_state,est_state,est_state_UAVs,robY,robX)
    % Display rhe true state of the environment with a custom colormap.
    surf_true = [true_state true_state(:,ENV_SIZE); true_state(ENV_SIZE,:) true_state(ENV_SIZE,ENV_SIZE)];
    colormap(get_colormap(200));
    figure(2); 
    surf(0:ENV_SIZE,0:ENV_SIZE,surf_true); view(0,90); caxis([0 1]);
    title("True State of Environment");

    % Plot est_state for each UAV on subplot respectively.
    % Create subplot to visualize individual UAV mapping. (MAX = 5 UAVs)
    colormap(get_colormap(200));
    figure(1);
    for i = 1:num_of_agents + 1
        subplot(2,3,i);
        if i == 1
            surf_est = [est_state est_state(:,ENV_SIZE); est_state(ENV_SIZE,:) est_state(ENV_SIZE,ENV_SIZE)];
            surf(0:ENV_SIZE,0:ENV_SIZE,surf_est); view(0,90); caxis([0 1]);
            title("Centralized Est. State");
        else
            surf_est_UAV = [est_state_UAVs(:,:,i-1) est_state_UAVs(:,ENV_SIZE); est_state_UAVs(ENV_SIZE,:,i-1) ...
                            est_state_UAVs(ENV_SIZE,ENV_SIZE)];
            surf(0:ENV_SIZE,0:ENV_SIZE,surf_est_UAV); view(0,90); caxis([0 1]);
            title(['UAV ' num2str(i-1)]); 
        end 
    end

    % Illustrate robot on est_state environment.
    subplot(2,3,1);
    hold on
    for t = 1:size(robX,2)
        text(robY(t)-1.1,robX(t)-0.3,2,'•');
    end
    hold off
end

% Finds the neighbors of a given square. Illegal neighbors are indicated
% with coordinates (-1,-1).
function path = expand_helper(i,j,ENV_SIZE)
    expand_order = ["left" "down" "right" "up"];
    path = zeros(3,2);
    index = 1;
    for idx = expand_order
        path(index,:) = expand(i,j,idx,ENV_SIZE);
        index = index + 1;
    end
end
function value = expand(i,j,val,ENV_SIZE)
    if val == "left" && j > 1
        value = [i,j-1];
    elseif val == "down" && i < ENV_SIZE
        value = [i+1,j];
    elseif val == "right" && j < ENV_SIZE
        value = [i,j+1];
    elseif val == "up" && i > 1
        value = [i-1,j];
    % At an invalid spot on the matrix
    else
        value = -1;
    end
end

