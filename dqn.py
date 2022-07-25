import random
import retro
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

class ReplayBuffer:
     def __init__(self):
        self.gameplay_experiences = deque(maxlen=10000)

    def store_gameplay_experience(self, state, next_state, reward,action, done):
        """stores a transition"""
        self.gameplay_experiences.append((state, next_state, reward,action, done))
        return

    def sample_gameplay_batch(self):
        """returns a list of gameplay experiences"""
        batch_size = 32
        sampled_gameplay_batch = random.sample(self.gameplay_experiences,k=batch_size) if len( self.gameplay_experiences) > batch_size
        else self.gameplay_experiences
            state_batch = []
            next_state_batch = []
            action_batch = []
            reward_batch = []
            done_batch = []
            for gameplay_experience in sampled_gameplay_batch:
                state_batch.append(gameplay_experience[0])
                next_state_batch.append(gameplay_experience[1])
                reward_batch.append(gameplay_experience[2])
                action_batch.append(gameplay_experience[3])
                done_batch.append(gameplay_experience[4])
            return np.array(state_batch), np.array(next_state_batch), np.array(action_batch), \
            np.array(reward_batch), np.array(done_batch)


class DQNAgent:
    def __init__(self):
        self.q_net = self.build_dqn_model()
        self.target_q_net = self.build_dqn_model()
    
    @staticmethod
    def build_dqn_model():
        """builds the dqn that predicts q values"""
        """input should have the shape of the state"""
        """output should have the shape as the action space for 1 q value per action"""
        q_net = Sequential()
        q_net.add(Dense(128, input_dim=57344, activation='relu',kernel_initializer='he_uniform'))
        q_net.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
        q_net.add(Dense(468, activation='linear',kernel_initializer='he_uniform'))
        q_net.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return q_net
    def random_policy(self):
        """outputs random action"""
        return np.random.randint(0, 468)

    def collect_policy(self, state, inx, iny):
        """some randomness"""
        if np.random.random() < 0.1:
            return self.random_policy()
        return self.policy(state, inx, iny)

    def policy(self, state, inx, iny):
        """takes a state from the env and returns an action"""
        """with the highest q value and should be taken in the next step"""
        state = cv2.resize(state, (inx, iny))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = np.reshape(state, (1, -1))
        print(state.shape)
        state_input = tf.convert_to_tensor(state[:], dtype=tf.float32)
        action_q = self.target_q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def update_target_network(self):
        """updates the current target_q_net"""
        self.target_q_net.set_weights(self.q_net.get_weights())

    def train(self, batch):
        """trains the network with a batch of gameplay experiences"""
        """to help predict the q values"""
        state_batch, next_state_batch, action_batch, \
        reward_batch, done_batch = batch
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for i in range(32):
            if done_batch[i]:
                target_q[i][0][action_batch[i]] = reward_batch[i]
            else:
                target_q[i][0][action_batch[i]] = reward_batch[i] + 0.95* max_next_q[i][action_batch[i]]
        training_history = self.q_net.fit(x=state_batch, y=target_q)
        loss = training_history.history['loss']
        return loss
    
    def collect_gameplay_experiences(env, agent, buffer, inx, iny):
        """collect experience and stores the gample exp in buffer"""
        state = env.reset()
        done = False
        realReward = 0
        counter = 0
        score = 0
        scoreTracker = 0
        coins = 0
        coinsTracker = 0
        yoshiCoins = 0
        yoshiCoinsTracker = 0
        xPosPrevious = 0
        yPosPrevious = 0
        checkpoint = False
        checkpointValue = 0
        endOfLevel = 0
        powerUps = 0
        powerUpsLast = 0
        jump = 0
        while not done:
            env.render()
            # action = agent.policy(state, inx, iny)
            action = agent.collect_policy(state, inx, iny)
            next_state, reward, done, info = env.step(action)
            score = info['score']
            coins = info['coins']
            yoshiCoins = info['yoshiCoins']
            dead = info['dead']
            xPos = info['x']
            yPos = info['y']
            jump = info['jump']
            checkpointValue = info['checkpoint']
            endOfLevel = info['endOfLevel']
            powerUps = info['powerups']

            if score > 0:
                if score > scoreTracker:
                    realReward = (score * 10)
                    scoreTracker = score
            if coins > 0:
                if coins > coinsTracker:
                    realReward += (coins - coinsTracker)
                    coinsTracker = coins

            if yoshiCoins > 0:
                if yoshiCoins > yoshiCoinsTracker:
                    realReward += (yoshiCoins - yoshiCoinsTracker) * 10
                    yoshiCoinsTracker = yoshiCoins

            if xPos > xPosPrevious:
                if jump > 0:
                    realReward += 0.1
                realReward += (xPos / 10000)
                xPosPrevious = xPos
                counter = 0
            else:
                counter += 1
                realReward -= 0.01

            if yPos < yPosPrevious:
                realReward += 0.1
                yPosPrevious = yPos
            elif yPos < yPosPrevious:
                yPosPrevious = yPos

            if powerUps == 0:
                if powerUpsLast == 1:
                    realReward -= 5
                    print("Lost Upgrade")
                elif powerUps == 1:
                    if powerUpsLast == 1 or powerUpsLast == 0:
                        realReward += 0.0025
                    elif powerUpsLast == 2:
                        realReward -= 50
                        print("Lost Upgrade")
            powerUpsLast = powerUps

            if checkpointValue == 1 and checkpoint == False:
                realReward += 200
                checkpoint = True

            if endOfLevel == 1:
                realReward += 1000000
                done = True
                return 1
            if counter == 1000:
                realReward -= 125
                done = True
            if dead == 0:
                realReward -= 100
                done = True
            """Flatenning the current state to avoid errors of q net input"""
            state = cv2.resize(state, (inx, iny))
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            state = np.reshape(state, (1, -1))
            """Flatenning the next state to avoid errors of q net input"""
            next_state_flat = next_state
            next_state_flat = cv2.resize(next_state_flat, (inx, iny))
            next_state_flat = cv2.cvtColor(next_state_flat,cv2.COLOR_BGR2GRAY)
            next_state_flat = np.reshape(next_state_flat, (1, -1))
            buffer.store_gameplay_experience(state, next_state_flat,realReward, action, done)
            state = next_state
        return 0

    def save_model(q_net, target_q_net, ep):
        q_net.save('winner_models\\q_net_model-episode{0}'.format(ep))
        target_q_net.save('winner_models\\target_q_net_modelepisode{0}'.format(ep))
        print("Model successfully saved :)")

    # here comes the training loop
    def train_model(max_episodes=10000):
        """trains the dqn agent"""
        agent = DQNAgent()
        buffer = ReplayBuffer()
        env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', use_restricted_actions=retro.Actions.DISCRETE)
        inx, iny, inc = env.observation_space.shape
        #Descomentar las lineas siguientes si se quiere cargar un modelo guardado
        #agent.q_net = tf.keras.models.load_model('winner_models\q_net_modelepisode697')
        #agent.target_q_net = tf.keras.models.load_model('winner_models\\target_q_net_modelepisode697')
        for episode_cnt in range(max_episodes):
            print("\nEpisode n°: {0}".format(episode_cnt))
            winner = collect_gameplay_experiences(env, agent, buffer, inx, iny)
            if winner == 1:
                save_model(q_net=agent.q_net, target_q_net=agent.target_q_net, ep=episode_cnt)
                env.close()
                return episode_cnt
            gameplay_experience_batch = buffer.sample_gameplay_batch()
            loss = agent.train(gameplay_experience_batch)
            if episode_cnt % 20 == 0:
                agent.update_target_network()
            print("Loss is {0}".format(loss))
    #Descomentar la linea siguiente si se quiere guardar el modelo al final de todas las iteraciones
    #save_model(q_net=agent.q_net, target_q_net=agent.target_q_net,ep=max_episodes)
    env.close()
    
episode_win = train_model()
print("winner on episode {0}".format(episode_win))
