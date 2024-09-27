clc,clear,close all
% test 092601
rng(0) % 随机数的可复现性

Nstate = 20;
Action = {'left', 'right'};
Epsilon = 0.9;
Alpha = 0.1;
Lambda = 0.9;
max_episode = 30;
freshtime = 0.3;


q_table = rl(Nstate, Action, max_episode, Lambda, Alpha, Epsilon);
% fprintf("\n Q-table: \n");
% disp(q_table);

function Qtable = create_q_table(Nstate, actions)
Qtable = zeros(Nstate, length(actions));
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
function [S_, R] = get_env_feedback(S, A, Nstate)
if A == 2
    if S == Nstate-1
        S_ = 'terminal';
        R = 1;
    else
        S_ = S+1;
        R = 0;
    end
else
    R = 0;
    if S == 1
        S_ = S;
    else
        S_ = S-1;
    end
end
end

% 更新环境并输出相对应的更新环境
function ultimate_step_counter = update_env(S, episode, step_counter, Nstate)
env_list =[repmat('-', 1, Nstate-1), 'T'] ;
% fprintf('%s\n', env_list);
if S == 'terminal'
    fprintf('第%d次循环的总步数是%d\n', episode, step_counter)
    ultimate_step_counter = step_counter;
else
    % fprintf('S:   %d  \n', S);
    env_list(S)='o';
    % fprintf('%s\n', env_list);
    ultimate_step_counter = false;
end
end

function q_table = rl(Nstate, Action, max_episode, Lambda, Alpha, Epsilon)

each_step_count = zeros(1, max_episode);
q_table = create_q_table(Nstate, Action);

for episode=1:max_episode
    step_counter = 0;
    S = 1;
    is_terminated = false;
    update_env(S, episode, step_counter, Nstate);
    while ~is_terminated
        A = choose_action(S, q_table, Epsilon, Action);
        fprintf('S\t A\n')
        fprintf('%d \t %d\n', S, A);
        q_predict = q_table(S, A);
        [S_, R] = get_env_feedback(S,A, Nstate);   % 进行状态更新，并返回下一步状态和奖励
        disp('episode')
        disp(episode)
        disp('Q_table')
        disp(q_table)
        if S_ ~= 'terminal'
            q_target = R+Lambda*max(q_table(S_,:));
        else
            q_target = R;
            is_terminated = true;
        end
        fprintf('q_target,\t q_predict\n')
        fprintf('%.6f \t %.6f\n', q_target, q_predict);
        q_table(S,A) = q_table(S,A)+Alpha*(q_target-q_predict);
        S = S_;

        step_note_control = update_env(S, episode, step_counter+1, Nstate);
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
