from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam




def train(replay_memory, grid, End_Flag, i , j):
        x_pos,y_pos = [], []
        
        if len(replay_memory) < 1000:
            return

        mini_batch = getRandomChildren(replay_memory, grid, i , j) # Returns at random children of current node from replay memory

        current_states = np.array([cur_states[0] for cur_states in mini_batch])
        current_Qs_list = model.predict(current_states) # Predict the current Q value of states

        new_current_states = np.array([new_states[3] for new_states in mini_batch]) # Get future states from minibatch, then query NN model for Q values
        Qs_Future_List = target_model.predict(new_current_states) # When using target network, query it, otherwise main network should be queried        


        for i, (current_state, action, reward, End_Flag) in enumerate(mini_batch):  # Now we enumerate our mini_batch
            if not End_Flag: # If not a terminal state, get new q from future states, otherwise set it to 0
                q_Max_Future = Manhattan_Distance() # Get the future Q for taking  a step from manhattan distance
                new_q = reward + DISCOUNT * q_Max_Future() # Add  the Q future val to current Q val decison with reward 
            else:
                new_q = reward

            current_qs = current_Qs_list[i]
            current_qs[action] = new_q

            x_pos.append(current_state)
            y_pos.append(current_qs)

        model.fit(np.array(x_pos), np.array(y_pos), batch_size=BATCH_SIZE, shuffle=False, verbose=0 if  End_Flag else None)
 
        # updating to determine if we want to update target_model yet
        if End_Flag:
            target_update_counter += 1

        if target_update_counter > 5:
            target_model.set_weights(model.get_weights())
            target_update_counter = 0


        epsilon = max(0.01, epsilon*0.99975) #Update Epsilon with a epsilon decay value

def create_dqn(learn_rate, input_dims, n_actions, conv_units, dense_units):
    model = Sequential([
                Conv2D(conv_units, (3,3), activation='relu', padding='same', input_shape=input_dims),
                Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                Flatten(),
                Dense(dense_units, activation='relu'),
                Dense(dense_units, activation='relu'),
                Dense(n_actions, activation='linear')])

    model.compile(optimizer=Adam(lr=learn_rate, epsilon=1e-4), loss='CrossEntropyLoss')

    return model