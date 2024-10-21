import numpy as np
from copy import deepcopy


def random_learning(
    net,
    loss_fun,
    max_its=1000,
    max_mutations=1000,
    step=1 / 2**5,
    threshold=1e-3,
    info=False,
):
    """
    desc: a function that generates a random pertubation p
    and test is the network + p and network - p offer an improvement.

    returns loss over iterations,
    (iterations reached,total mutation attempts)
    """
    losses = [loss_fun(net)]
    total_k = 0

    for i in range(max_its):
        if info:
            print("Iter", i, "Loss", losses[-1])

        if losses[-1] < threshold:
            if info:
                print("Stopped Due to Threshold Reached")
            break

        for k in range(max_mutations):  # mutate so many times before giving up
            if info:
                print(f"Mutation: {k}/max_mutations",end = "\r")
            pertubations = random_mutate(net, step)
            loss = loss_fun(net)

            if loss < losses[-1]:
                losses.append(loss)
                break
            else:
                # pertubate in the opposite direction to save generating another pertubation
                undo_mutate(net, pertubations)
                undo_mutate(net, pertubations)

            loss = loss_fun(net)
            if loss < losses[-1]:
                losses.append(loss)
                break
            else:
                make_mutate(net, pertubations)

        total_k += k
        if k == max_mutations - 1:
            if info:
                print("Stopped Due to Max Mutations Reached")
            break

    return losses, (i, total_k)

def random_learning2(
    net,
    loss_fun,
    max_its=1000,
    max_mutations=1000,
    step= 2**-5,
    threshold=1e-3,
    eps = 2**-31,
    info=False,
):
    """
    desc: a function that takes a very small value eps and pertubates the network
    this gives an approximation to the derivative and adjust the net by that times the step.

    returns loss over iterations,
    (iterations reached,total mutation attempts)
    """
    losses = [loss_fun(net)]
    total_k = 0

    for i in range(max_its):
        if info:
            print("Iter", i, "Loss", losses[-1])

        if losses[-1] < threshold:
            if info:
                print("Stopped Due to Threshold Reached")
            break

        for k in range(max_mutations):  # mutate so many times before giving up
            if info:
                print(f"Mutation: {k}/max_mutations",end = "\r")
            pertubations, scale = random_mutate_norm(net, eps)
            loss = loss_fun(net)
            undo_mutate(net,pertubations)

            if loss < losses[-1]:
                make_mutate(net,p_multiply(pertubations,step/scale))
                losses.append(loss_fun(net))
                break
            else: # do nothing
                pass

        total_k += k
        if k == max_mutations - 1:
            if info:
                print("Stopped Due to Max Mutations Reached")
            break

    return losses, (i, total_k)


def random_mutate(net, step: float):  # randomly mutate the network
    pertubations = []
    for index in net.mutateable_layers:
        w_shape = np.shape(net.layers[index].weights)
        b_shape = np.shape(net.layers[index].biases)

        w_pertubation = step * np.random.normal(size=w_shape)
        b_pertubation = step * np.random.normal(size=b_shape)

        pertubations.append([w_pertubation, b_pertubation])

        net.layers[index].weights += w_pertubation
        net.layers[index].biases += b_pertubation

    return pertubations  # make the inside arrays?

def random_mutate_norm(net, step: float):  # randomly mutate the network
    pertubations = []
    total_size = 0
    total_abs_sum = 0
    for index in net.mutateable_layers:
        w_shape = np.shape(net.layers[index].weights)
        b_shape = np.shape(net.layers[index].biases)

        w_pertubation = np.random.normal(size=w_shape, scale=step)
        b_pertubation = np.random.normal(size=b_shape, scale=step)

        pertubations.append([w_pertubation, b_pertubation])
        total_size += w_pertubation.size + b_pertubation.size
        total_abs_sum += np.sum(np.abs(w_pertubation)) + np.sum(np.abs(w_pertubation)) 

        net.layers[index].weights += w_pertubation
        net.layers[index].biases += b_pertubation

    return pertubations , total_abs_sum / total_size # make the inside arrays?


def undo_mutate(net, pertubations):
    i = 0
    for index in net.mutateable_layers:
        w_pertubation = pertubations[i][0]
        b_pertubation = pertubations[i][1]

        net.layers[index].weights -= w_pertubation
        net.layers[index].biases -= b_pertubation

        i += 1


def make_mutate(net, pertubations):
    i = 0
    for index in net.mutateable_layers:
        w_pertubation = pertubations[i][0]
        b_pertubation = pertubations[i][1]

        net.layers[index].weights += w_pertubation
        net.layers[index].biases += b_pertubation

        i += 1


def p_add(p1, p2):
    """
    adds p2 to the p1 pertubation
    """
    for i in range(len(p1)):
        for j in range(len(p1[i])):
            p1[i][j] += p2[i][j]

    return p1


def p_multiply(p, n):
    """
    multiplies pertubation by n
    """
    for i in range(len(p)):
        for j in range(len(p[i])):
            p[i][j] = p[i][j] * n

    return p


def random_learning_momentum(
    net,
    loss_fun,
    max_its=1000,
    max_mutations=1000,
    step=1 / 2**5,
    threshold=1e-3,
    momentum=0.25,
    info=False,
):
    """
    returns loss over iterations,
    (iterations reached,total mutation attempts)
    """
    losses = [loss_fun(net)]
    total_k = 0

    for i in range(max_its):
        if info:
            print("Iter", i, "Loss", losses[-1])

        if losses[-1] < threshold:
            if info:
                print("Stopped Due to Threshold Reached")
            break

        if i != 0:  # should this go after the next vector is found?
            velocity = p_multiply(old_pertubation, momentum)
            make_mutate(net, velocity)
            losses.append(loss_fun(net))

        for k in range(max_mutations):  # mutate so many times before giving up
            pertubations = random_mutate(net, step)

            loss = loss_fun(net)

            if loss < losses[-1]:
                losses[-1] = loss

                if i != 0:
                    old_pertubation = p_add(pertubations, velocity)
                else:
                    old_pertubation = pertubations

                break
            else:
                # pertubate in the opposite direction to save generating another pertubation
                undo_mutate(net, pertubations)
                undo_mutate(net, pertubations)

            loss = loss_fun(net)
            if loss < losses[-1]:
                losses[-1] = loss
                if i != 0:
                    old_pertubation = p_add(p_multiply(pertubations, -1), velocity)
                else:
                    old_pertubation = p_multiply(pertubations, -1)

                break
            else:
                make_mutate(net, pertubations)

        total_k += k
        if k == max_mutations - 1:
            if info:
                print("Stopped Due to Max Mutations Reached")
            break

    return losses, (i, total_k)
