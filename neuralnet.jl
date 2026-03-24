function train(problem::ExcursionProblem,batch_size::Int,epochs::Int);
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    # Construct the layer
    model = Chain(Dense(2, 64, tanh), Chain(Dense(64, 64, tanh), Chain(Dense(64, 32, tanh), Dense(32, 1, sigmoid)))) #input state,output action probs

    #dev = reactant_device() -- this needs to have all data 32 bit?
    dev = cpu_device()
    # Parameter and State Variables
    ps, st = Lux.setup(rng, model) |> dev

    ## First construct a TrainState
    train_state = Training.TrainState(model, ps, st, Adam(0.001f0))

    average_returns =[]
    s0 = starting_state(problem)
    traj = Trajectory() 
    T = problem.trajectory_length
    for _ in 1:epochs
        states_list  = Matrix{Float32}[]
        actions_list = Matrix{Float32}[]
        returns_list = Matrix{Float32}[]
        tot_returns = 0.0
        for _ in 1:batch_size
            sample_trajectory!(traj, problem, s0, model, ps, st)
            pass = transitions(traj, problem.γ)
            tot_returns +=  pass[1][5]
            push!(states_list,  Float32.(hcat([collect(p[1]) for p in pass]...)))  # (2, T)
            push!(actions_list, Float32.(reshape([p[2] for p in pass], 1, T)))     # (1, T)
            push!(returns_list, Float32.(reshape([p[5] for p in pass], 1, T)))     # (1, T)
        end
        # hcat across batch: (2, T*batch) and (1, T*batch)
        states  = hcat(states_list...)  |> dev
        actions = hcat(actions_list...) |> dev
        returns = hcat(returns_list...) |> dev
        #accumulate gradient via NN
    # Both these steps can be combined into a single call (preferred approach)
        gs, loss, stats, train_state = Training.single_train_step!(
            AutoEnzyme(), #auto diff
            PGLoss, #loss function
            (states,actions,returns), #input/target pair
            train_state #model params etc
        )
        avg_return = tot_returns/batch_size
        push!(average_returns,avg_return)
    end
    return  average_returns
end

function PGLoss(model, ps, st, (states, actions, returns)) #acts on whole traj at once or single element?
    action_probs, st = Lux.apply(model,states,ps,st) #get p up for the state
    one_log_probs = log.(clamp.(action_probs,1f-7,1f0))
    zero_log_probs = log.(clamp.(1f0 .- action_probs, 1f-7, 1f0)) #clamp for numerical stability
    #change from 1/2 to 0/1
    a_binary = actions .- 1
    selected_log_probs = a_binary .* one_log_probs .+ (1f0 .- a_binary) .* zero_log_probs #none action term dissapears
    loss = -mean(selected_log_probs.*returns)
    return loss, st, (;)
end


function sample_trajectory!(traj::Trajectory, problem::ExcursionProblem, s0::Tuple{Int64,Int64},model, ps, st)
    empty!(traj.states)
    empty!(traj.actions)
    empty!(traj.rewards)
    push!(traj.states, s0)

    while true
        current_state = traj.states[end]
        if is_terminal(problem, current_state)
            break
        end
        action, p_up = sample_action(model,current_state,ps,st)
        ns = next_state(problem, current_state, action)
        p_action = action == 2 ? p_up : 1-p_up
        # 0.5 is the uniform probability
        r = reward(problem, ns) - log(p_action/0.5)
        append_transition(traj, ns, action, r)
    end
end

function sample_action(model,state,ps,st)
    s_input = Float32.(reshape(collect(state),2,1))
    output, st = Lux.apply(model,s_input,ps,st)
    p_up = output[1]
    action = rand() < p_up ? 2 : 1
    return action, p_up    
end


