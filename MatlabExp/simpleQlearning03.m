clc,clear,close all
% test 092601
% rng(0) % 随机数的可复现性

Nstate = 4; % 状态为一个4*4的方格。
Action = {'left', 'right', 'up', 'down'};
Epsilon = 0.9;
Alpha = 0.1;
Lambda = 0.9;
max_episode = 500;
freshtime = 0.3;

q_table = rl(Nstate, Action, max_episode, Lambda, Alpha, Epsilon);
% fprintf("\n Q-table: \n");
% disp(q_table);

function Qtable = create_q_table(Nstate, actions)
Qtable = zeros(Nstate^2, length(actions));
end

% 刻画代理者选择的动作
function action_index = choose_action(state, q_table, Epsilon, Action)
state_actions = q_table(state, :);
if rand>Epsilon|| all(state_actions==0)
    action_index = randi([1, length(Action)]);
else
    [~, maxindex] = max(state_actions);
    action_index = maxindex;
end
end

% 刻画代理者选择的动作对当前状态的影响
% Action = {'left', 'right', 'up', 'down'};
function [Sx_, Sy_, R] = get_env_feedback(Sx, Sy, A, Nstate)
if A == 1
    disp('A=1')
    if Sx==2 && Sy==4
        Sx_ = 'trap';
        Sy_ = 'trap';
        R = -1;
    elseif Sx==3 && Sy==4
        Sx_ = 'terminal';
        Sy_ = 'terminal';
        R = 1;
    elseif Sy == 1
        Sx_ = Sx;
        Sy_ = Sy;
        R = 0;
    else
        Sx_ = Sx;
        Sy_ = Sy - 1;
        R = 0;
    end
elseif  A == 2
    disp('A=2')
    if (Sx==2 && Sy==2) || (Sx==3 && Sy==1)
        Sx_ = 'trap';
        Sy_ = 'trap';
        R = -1;
    elseif Sy == Nstate
        Sx_ = Sx;
        Sy_ = Sy;
        R = 0;
    else
        Sx_ = Sx;
        Sy_ = Sy + 1;
        R = 0;
    end
elseif  A == 3
    disp('A=3')
    if Sx==4 && Sy==2
        Sx_ = 'trap';
        Sy_ = 'trap';
        R = -1;
    elseif Sx==4 && Sy==3
        Sx_ = 'terminal';
        Sy_ = 'terminal';
        R = 1;
    elseif Sx == 1
        Sx_ = Sx;
        Sy_ = Sy;
        R = 0;
    else
        Sx_ = Sx - 1;
        Sy_ = Sy;
        R = 0;
    end
elseif  A == 4
    disp('A=4')
    if (Sx==2 && Sy==2) || (Sx==1 && Sy==3)
        Sx_ = 'trap';
        Sy_ = 'trap';
        R = -1;
    elseif Sx == Nstate
        Sx_ = Sx;
        Sy_ = Sy;
        R = 0;
    else
        Sx_ = Sx + 1;
        Sy_ = Sy;
        R = 0;
    end
end
end

% 更新环境并输出相对应的更新环境
function ultimate_step_counter = update_env(Sx, Sy, episode, step_counter, Nstate)
env_list =[repmat('-', Nstate, Nstate)] ;   % 状态有16个，并且对于特殊的状态，需要特别标记
env_list(3,3) = 'x';
env_list(3,2) = 'o';
env_list(2,3) = 'o';

if strcmp(Sx ,'terminal') && strcmp(Sy ,'terminal')
    fprintf('第%d次循环成功，总步数是%d\n', episode, step_counter)
    ultimate_step_counter = step_counter;
elseif strcmp(Sx ,'trap') && strcmp(Sy ,'trap')
    fprintf('第%d次循环失败，总步数是%d\n', episode, step_counter)
    ultimate_step_counter = step_counter;
else
    % fprintf('S:   %d  \n', S);
    env_list(Sx, Sy)='*';
    % fprintf('%s\n', env_list);
    for i = 1:Nstate
        fprintf('%s\n', env_list(i,:));
    end
    ultimate_step_counter = false;
end
end

function q_table = rl(Nstate, Action, max_episode, Lambda, Alpha, Epsilon)

each_step_count = zeros(1, max_episode);
q_table = create_q_table(Nstate, Action);

for episode=1:max_episode
    step_counter = 0;
    Sx = 1;
    Sy = 1;
    is_terminated = false;
    update_env(Sx, Sy, episode, step_counter, Nstate);
    while ~is_terminated
        fprintf('Sx: %d \t Sy: %d\n', Sx, Sy);
        A = choose_action(Sx+(Sy-1)*Nstate, q_table, Epsilon, Action);
        fprintf('Sx\t Sy\t A\n')
        fprintf('Sx: %d \t Sy: %d\t %d\n', Sx, Sy, A);
        q_predict = q_table(Sx+(Sy-1)*Nstate, A);
        [Sx_, Sy_, R] = get_env_feedback(Sx, Sy, A, Nstate);   % 进行状态更新，并返回下一步状态和奖励
        fprintf('Sx_: %s \t Sy_: %s\t %d\n', Sx_, Sy_, A);
        disp('episode')
        disp(episode)
        disp('Q_table')
        disp(q_table)
        if ~strcmp(Sx_ ,'terminal') && ~strcmp(Sy_ ,'terminal') && ~strcmp(Sx_ ,'trap') && ~strcmp(Sy_ ,'trap')
            q_target = R+Lambda*max(q_table(Sx_+(Sy_-1)*Nstate,:));
        else
            q_target = R;
            is_terminated = true;
        end
        fprintf('q_target,\t q_predict\n')
        fprintf('%.6f \t %.6f\n', q_target, q_predict);
        q_table(Sx+(Sy-1)*Nstate,A) = q_table(Sx+(Sy-1)*Nstate,A)+Alpha*(q_target-q_predict);
        Sx = Sx_;
        Sy = Sy_;
        if ~strcmp(Sx ,'terminal') && ~strcmp(Sy ,'terminal') && ~strcmp(Sx ,'trap') && ~strcmp(Sy ,'trap')
            fprintf('Sx: %d \t Sy: %d\n', Sx, Sy);
        else
            fprintf('Sx: %s \t Sy: %s\n', Sx, Sy);
        end
        step_note_control = update_env(Sx, Sy, episode, step_counter+1, Nstate);
        if step_note_control~=false
            each_step_count(episode)= step_note_control;
        end
        step_counter = step_counter+1;
    end

end
plot(1:length(each_step_count), each_step_count, 'r-o')
xlabel('运行次数')
ylabel('步数')
title('Q学习效果')
end
