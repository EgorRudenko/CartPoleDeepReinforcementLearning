import asyncio
import websockets
import numpy as np
import pickle

#plt.axis([0, 10, 0, 1])
#rng = np.random.default_rng(seed=5)
rng = np.random  # could be used to define specific seed


decay_rate = 0.9        # how fast previous changes in weight become irrelevant (bigger - slower)
gamma = 0.99    # importance of long term success over short term
toLearn = False
to_load = True
saveFrequency = 500
batch_size = 10             # update weights every 10 iterations
learning_rate = 1e-4
D: int = 3  # input layer neurons number
H: int = 150  # first hidden layer neurons number
H1: int = 100   # second hidden layer neurons number
H2: int = 50    # third hidden layer neurons number


def sigmoid(x):                         # activation function on the output layer
    return 1.0 / (1.0 + np.exp(-x))


def policy_forward(x, model):
    h = np.dot(model["W1"], x)  # values on hidden layer
    # Apply non-linearity with ReLU.
    h[h < 0] *= 0.1  # Numpy feature for every element which satisfies h<0 multiply it by .1 (leaky ReLU)
    h1 = np.dot(model["W2"], h)
    h1[h1 < 0] *= 0.1
    h2 = np.dot(model["W3"], h1)
    h2[h2 < 0] *= 0.1
    logit = np.dot(model["W4"], h2)
    # Apply the sigmoid function (non-linear activation).
    p = sigmoid(logit)  # actual decision
    # and the hidden "state" that you need for backpropagation.
    return p, [h, h1, h2]


def policy_backward(eph0, eph1, eph2, epd, model, epx):
    # epd is results given by network over epoch times the discounted reward  "Expected Present Discounted Return"
    # eph is the list of states of hidden layers "Hidden States History"
    # epx are observations

    # normal explanation of how it works: https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html

    # we define empty delta w's
    dw4 = np.zeros_like(model["W4"])
    dw3 = np.zeros_like(model["W3"])
    dw2 = np.zeros_like(model["W2"])
    dw1 = np.zeros_like(model["W1"])
    for i in range(len(eph0)):      # iterating through actions (not only in one episode, but all saved from the batch)
        dw4 += eph2[i] * epd[i]  # change in hidden => output weights (derivative of cost function times inputs)
        # little clarification: there is no "cost" function here. It is achieved by using some other methods
        # however epd is equivalent of such loss function

        dh1 = np.dot(epd[i], model["W4"])  # importance of each single hidden neuron on the last hidden layer for error
        dh1[eph2[i] <= 0] *= 0.1  # multiply by the derivative of ReLU
        dw3 += np.dot(np.array([dh1]).T, [eph1[i]]) # error on the last hidden layer times inputs to get error on w's

        dh2 = np.dot(dh1, model["W3"])
        dh2[eph1[i] <= 0] *= 0.1
        dw2 += np.dot(np.array([dh2]).T, [eph0[i]])

        dh3 = np.dot(dh2, model["W2"])
        dh3[eph0[i] <= 0] *= 0.1
        dw1 += np.dot(np.array([dh3]).T, [epx[i]])

    return {"W1": dw1, "W2": dw2, "W3": dw3, "W4": dw4}


# All preprocessed observations for the episode.
xs = []
# All hidden "states" (from the network) for the episode. Basically values on hidden neurons from forward propagation
hs0 = []
hs1 = []
hs2 = []
# All gradients of probability of actions (with respect to observations) for the episode.
dlogps: list = []
# All rewards for the episode.
drs = []
episode_number = 0      # what episode are we playing

def discount_rewards(r, gamma):  # r are rewards
    # in this function we want to make sequences of good actions more important than just good actions
    discounted_r = np.zeros_like(r)  # matrix of form r filled with zeroes
    running_add = 0
    # From the last reward to the first...
    for t in reversed(range(0, r.size)):
        # ...compute the discounted reward
        if r[t] == 0:
            running_add = 0
        running_add = running_add * gamma + r[t]  # if there are consecutive rewards make this reward better
        discounted_r[t] = running_add
    return discounted_r

running_reward = None
reward_sum = 0

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_out, fan_in))


model = {
    "W1": xavier_init(D, H),  # Input => hidden weights
    "W2": xavier_init(H, H1),
    "W3": xavier_init(H1, H2),
    "W4": xavier_init(H2, 1)
}


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

try:
    num = np.load("num.npy")
except:
    num = 0

if to_load:
    model = load_dict(f'model{num-1}')

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}


def ai(input_layer):
    reward = 0
    x = np.array(input_layer[0:3])
    is_game_over = input_layer[3]

    aprob, h = policy_forward(x, model)
    aprob = aprob[0]
    action = 1 if rng.uniform() < aprob else 0

    global xs, hs0, hs1, hs2, dlogps, reward_sum, drs, episode_number, running_reward, eph, num, toLearn

    xs.append(x)
    hs0.append(h[0])
    hs1.append(h[1])
    hs2.append(h[2])

    y = 1 if action == 1 else 0

    dlogps.append(y - aprob)
    if not is_game_over:
        reward = 0.1
    else:
        reward = 0
    reward_sum += reward
    # 10. Append the reward for the previous action.
    drs.append(reward)

    if not is_game_over:
        return str(-1 if action == 0 else 1)
    else:
        if toLearn:
            episode_number += 1
            # - Observation frames (inputs),
            epx = np.vstack(xs)
            # - hidden "states" (from the network),
            eph0 = np.vstack(hs0)
            eph1 = np.vstack(hs1)
            eph2 = np.vstack(hs2)
            # - gradients of action log probabilities,
            epdlogp = np.vstack(dlogps)
            # - and received rewards for the past episode.
            epr = np.vstack(drs)

            # Reset the stored variables for the new episode:
            xs = []
            hs0 = []
            hs1 = []
            hs2 = []
            dlogps = []
            drs = []

            discounted_epr = discount_rewards(epr, gamma)
            discounted_epr -= np.mean(discounted_epr)
            if np.std(discounted_epr) != 0:
                discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr
            grad = policy_backward(eph0, eph1, eph2, epdlogp, model, epx)

            for k in model:
                grad_buffer[k] += grad[k]

            if episode_number % batch_size == 0:
                # explanation: https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html#rmsprop
                print("NewIteration")
                for k, v in model.items():
                    # The gradient.
                    g = grad_buffer[k]                      # found changes in w's while backpropagation
                    # Use the RMSProp discounting factor.
                    rmsprop_cache[k] = (
                            decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                            # all the previous changes are important + new in some little amount to get smooth result
                    )
                    # Update the policy network with a learning rate
                    # and the RMSProp optimizer using gradient ascent (not decent)
                    # (hence, there's no negative sign)
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    # Reset the gradient buffer at the end.
                    grad_buffer[k] = np.zeros_like(v)
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )
            print(
                "Resetting the Shit environment. Episode: {} Episode total reward: {} Running mean: {}".format(
                    episode_number, reward_sum, running_reward
                )
            )

            reward_sum = 0
            if episode_number % saveFrequency == 0:
                save_dict(model, f'model{num}')
                num += 1
                np.save('num', num)
        return str(0)


async def server(ws, path):
    async for msg in ws:
        response = ai(list(map(float, msg.split(" "))))
        await ws.send(response)


start_server = websockets.serve(server, "localhost", 5000)
#print(ai([1, 1, 1, 1, 1,1]))
print("server started")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
