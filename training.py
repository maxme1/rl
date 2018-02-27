def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def actor_critic(env: gym.Env, agent: ActorCritic, optimizer, n_steps, max_steps=None,
                 gamma=1, max_grad=50, alpha=.01):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    is_cuda = next(agent.model.parameters()).is_cuda
    while not done:
        # syncronization
        # agent.model.load_state_dict(shared_model.state_dict())

        values, rewards, logs, entropies = [], [], [], []
        for _ in range(n_steps):
            prob, value = agent.both(state)
            log_prob = torch.log(prob)
            entropy = - (log_prob * prob).sum(1).view(-1, 1)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, _ = env.step(action.cpu().numpy()[0, 0])
            total_reward = total_reward * gamma + reward

            values.append(value)
            rewards.append(reward)
            logs.append(log_prob)
            entropies.append(entropy)

            steps += 1
            done = done or (max_steps is not None and max_steps <= steps)
            if done:
                break

        if done:
            R = torch.zeros(1, 1)
            if is_cuda:
                R = R.cuda()
        else:
            R = agent.value(state).data

        values.append(Variable(R))
        val_loss = 0
        pol_loss = 0
        advantage = 0

        R = Variable(R)
        for i in range(len(rewards) - 1, -1, -1):
            R = R * gamma + rewards[i]
            diff = R - values[i]
            delta = gamma * values[i + 1] - values[i] + rewards[i]
            advantage = advantage * gamma + delta.data

            val_loss = val_loss + delta ** 2
            pol_loss = pol_loss - logs[i] * Variable(advantage) - entropies[i] * alpha

        optimizer.zero_grad()
        (pol_loss + val_loss).backward()
        torch.nn.utils.clip_grad_norm(agent.model.parameters(), max_grad)
        # share grads
        # ensure_shared_grads(agent.model, shared_model)
        optimizer.step()

    return total_reward


def sarsa(env: gym.Env, agent: QFunction, optimizer, gamma=1, logger=None):
    s = env.reset()
    steps = 0
    done = False
    a = agent.action(s)
    while not done:
        s_new, r, done, _ = env.step(a)
        if done:
            a_new = a
            next_val = 0
        else:
            a_new = agent.action(s_new)
            next_val = agent.value(s_new, a_new)

        value = agent.value(s, a)
        loss = ((r + gamma * next_val - value) ** 2).mean()
        if logger is not None:
            logger(loss.cpu().data.numpy().flatten()[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        s = s_new
        a = a_new
        steps += 1


def list_to_tensor(a):
    a = np.asarray(a)[None]
    a = torch.cat(torch.from_numpy(a))
    a = Variable(a[:, None]).cuda()
    return a


def q_update(agent, s, a, r, s_new, done, gamma):
    a = list_to_tensor(a)
    r = list_to_tensor(r).float().squeeze()
    not_done = ~np.asarray(done)
    not_done = np.where(not_done)[0]
    s = np.asarray(s)
    s_new = np.asarray(s_new)

    next_vals = Variable(torch.zeros(len(done))).cuda()
    values = agent.values(s)

    if np.prod(not_done.shape):
        non_final = s_new[not_done]
        not_done = list_to_tensor(not_done).squeeze()
        next_vals[not_done] = agent.values(non_final).max(dim=-1)[0]

    values = values.gather(1, a).squeeze()
    loss = ((r + gamma * next_vals - values) ** 2)

    return loss


def q_learning(env: gym.Env, agent: QFunction, optimizer, memory, batch_size, max_size, gamma=1):
    s = env.reset()
    steps = 0
    done = False
    total_reward = 0
    while not done:
        a = agent.action(s)
        s_new, r, done, _ = env.step(a)
        pack = s, a, r, s_new, done

        loss = q_update(agent, *zip(*[pack]), gamma).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(memory) < max_size:
            memory.append(pack)
        else:
            idx = np.random.randint(0, len(memory))
            memory[idx] = pack

        total_reward = total_reward * gamma + r
        s = s_new
        steps += 1

        if len(memory) > batch_size:
            data = []
            for idx in np.random.randint(0, len(memory), batch_size):
                data.append(memory[idx])

            loss = q_update(agent, *zip(*data), gamma).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return total_reward
