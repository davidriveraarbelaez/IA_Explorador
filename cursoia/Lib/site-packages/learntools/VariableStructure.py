from learntools import Network, Learning
import numpy as np
from copy import deepcopy


def VariableStructure(
    n_in: int,
    n_out: int,
    loss_fun,
    act_layer,
    epochs=100,
    max_its=100,
    max_mutations=100,
    depth_add_prob=0.1,
    mutation_rate=0.2,
    threshold=1e-3,
    reset=0,
    eps=1e-10,
    info=False,
    net=None,
):  # max_mutations should be max pertubations?
    if net == None:
        net = Network.network(n_in, n_out)
        net.add_layer(Network.layer_dense(n_in, n_out))

        net.random_initilisation()

    total_loss = []

    best_loss = np.inf
    last_loss = np.inf

    for epoch in range(epochs):  # go through the epochs
        # learn until it stops
        loss, (its, total_mutations) = Learning.random_learning(
            net, loss_fun, max_its, max_mutations, threshold=threshold
        )
        # loss
        total_loss += loss
        if info: #print 
            print(f"Epoch-{epoch}-loss-{total_loss[-1]}")


        if total_loss[-1] + eps < best_loss:
            #assign best net
            best_net = deepcopy(net)
            best_loss = total_loss[-1]
        
        if total_loss[-1] < threshold: # if loss reaches threshold
            break

        rem_its = max_its - its
        mutation_prob = rem_its / max_its - (1 - mutation_rate)

        if (
            np.random.uniform(0, 1) < mutation_prob
            and total_loss[-1] - last_loss < eps # we see a 0.1% decrease in loss
        ):
            # change the structure
            if (
                np.random.uniform(0, 1) < depth_add_prob
                or len(net.mutateable_layers) == 1
            ):  # add layer with depth_add prob or if we only have 1 layer
                if info:
                    print("Layer Added")

                net.add_layer(act_layer())
                new_layer = Network.layer_dense(net.n_out, net.n_out)
                new_layer.weights += 1 / net.n_out #make sure output is unaffected
                # print(new_layer.weights)
                net.add_layer(new_layer)

            else:
                layers = net.layers[:]
                mutate_loc = np.random.choice(
                    list(range(len(net.mutateable_layers[:-1])))
                )  # exclude last one as we edit n_out
                mutate_index = net.mutateable_layers[mutate_loc]

                old_layer = layers[mutate_index]
                new_layer = Network.layer_dense(old_layer.n_in, old_layer.n_out + 1)
                new_layer.weights[
                    : old_layer.weights.shape[0], : old_layer.weights.shape[1]
                ] = old_layer.weights[:]
                new_layer.biases[: old_layer.biases.shape[0]] = old_layer.biases[:]
                layers[mutate_index] = new_layer

                if info:
                    print(f"Node added in layer {mutate_index}")

                mutate_index = net.mutateable_layers[mutate_loc + 1]
                old_layer = layers[mutate_index]
                new_layer = Network.layer_dense(old_layer.n_in + 1, old_layer.n_out)
                new_layer.weights[
                    : old_layer.weights.shape[0], : old_layer.weights.shape[1]
                ] = old_layer.weights[:]
                new_layer.biases[: old_layer.biases.shape[0]] = old_layer.biases[:]
                layers[mutate_index] = new_layer

                net.layers = layers

            if net.check_integrity() == False:
                raise Exception("Failed Integrity")

            if reset == 1:
                net.reset()
            elif reset == 2:
                net.random_initilisation()

        last_loss = total_loss[-1]

    return best_net, total_loss
