def make_state(model,next_state):

    new_c = []
    new_h = []
    for i in range(model.num_layers):
        new_c.append(next_state[i][0])
        new_h.append(next_state[i][1])
    return new_c,new_h