clc,clear, close all
% 本代码的案例来自 https://www.bilibili.com/video/BV1Vh411N7F8
quality = 1000*ones(7);
reward = [-1, -1, -1, 0, -1, -1, -1;
    -1, -1, 0, -1, -1, -1, -1;
    -1, 0, -1, 0, -1, 0, -1;
    0, -1, 0, -1, 0, -1, -1;
    -1, -1, -1, 0, -1, 0, 100;
    -1, -1, 0, -1, 0, -1, 100;
    -1, -1, -1, -1, 0, 0, 0];
times = zeros(7);
gamma = 0.8;

for i = 1:1000
    total_reward = 0;
    state = randi([1,7]);
    ori_s = state;
    % disp(['state:', num2str(state), '  i: ', num2str(i)])
    s_s = state;
    
    %% 计算奖励函数
    while state~=7
        r_s = find(reward(state, :)>=0);  % 当前状态所能选择的下一步的库
        % disp(['r_s:   ', num2str(r_s)])

        s = r_s(randi([1, length(r_s)]));% 从下一步的库中选择下一步状态
        % disp(['s:   ', num2str(s)])

        s_s = [s_s, s];
        % disp('s_s')
        % disp(s_s)

        total_reward = total_reward+reward(state, s); % 计算奖励
        % disp('total_reward')
        % disp(total_reward)
        times(state, s) = times(state, s)-1;
        quality(state, s) = reward(state, s)+gamma*max(quality(s,:))+times(state, s);
        state = s; % 定义下一个状态
    end 
end
%% 根据奖励矩阵,随机生成初始状态，并生成奖励最大化的路径

% disp(quality)
quality(quality==1000)=quality(quality==1000)-(1000-min(min(quality))+1);
quality=quality-min(min(quality));
disp('quality')
disp(quality)

state = randi([1,7]);
fprintf('state: %d\n', state-1)
n_c = 0;
while state~=7
    [~, q_max_l] = max(quality(state, :));   % 本行中价值最大的动作
    q_action = find(quality(state, :)~=0); % 本行中可供转移的下一个动作
    n_s = q_max_l;
    save_r = rand;
    if save_r>0.6
    n_s = q_action(randi([1, length(q_action)]));
    end
    fprintf('随机数save_r的值是%.6f\n', save_r)
    state = n_s;
    disp(state-1)
    n_c = n_c+1;
    if n_c>5

        disp('本次求解归于失败')
        n=0;
        break
    end
end
