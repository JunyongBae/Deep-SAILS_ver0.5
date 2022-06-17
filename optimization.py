import os

from module import *

def algorithm(score, trial):
    dir = './score_'+ str(score) + '_trial_' + str(trial) + '/'
    os.makedirs(dir, exist_ok=True)

    # Branch definition
    axis, input_para = load_axis('brach_info.dat')
    all_branches = load_branches(axis)

    critic = Critic()

    # Load model
    model = construct_model(input_para)

    # Basic case simulation
    X, Y = basic_case(axis, all_branches, random = 0)

    # Basic case training
    model = train_model(model, X, Y, validation=False, dir = dir, label = 0)

    # Critic : progress check + new_visiting branches check
    visiting, _, _= critic.record(all_branches, model, X, label=0, dir = dir)

    # count visited branches
    num_of_visited_branches = np.sum(visiting)

    termination = False
    iteration = 0
    while not(termination):
        # Find uncertainty
        uncertainty = all_branch_uncertainty(model, all_branches, mc=500)

        # Find standard score of 1478K and select random 10 grey points in +-1.95
        idx, result = standard_score_of_1478(uncertainty, n = 1680, score = score, label=iteration, dir = dir)

        # Simulation and add to training data
        X, Y = simulation(idx, all_branches, visiting, X, Y)

        # Training
        model = train_model(model, X, Y, validation=False, label=iteration+1, dir = dir)

        # Critic : progress check + new_visiting branches check
        visiting, _, _  = critic.record(all_branches, model, X, label=iteration+1, dir = dir)

        # Termination Check
        if (np.sum(visiting) -  num_of_visited_branches) <= 1680 * 0.1:
            print('\n New visiting points : ' + str((np.sum(visiting) -  num_of_visited_branches)))
            termination = True
        else:
            iteration += 1
            print('\n New visiting points : ' + str((np.sum(visiting) -  num_of_visited_branches)))
            num_of_visited_branches = np.sum(visiting)

    # Find uncertainty
    uncertainty = all_branch_uncertainty(model, all_branches, mc=500)

    # Find standard score of 1478K and select random 10 grey points in +-1.95
    idx, result = standard_score_of_1478_all(uncertainty, score = score, label = iteration + 1, dir = dir)

    # Simulation and add to training data
    X, Y = simulation(idx, all_branches, visiting, X, Y)

    # Training
    model = train_model(model, X, Y, validation=False, label=iteration + 2, dir = dir)
    model = train_model(model, X, Y, validation=False, label=iteration + 2, dir = dir)
    model = train_model(model, X, Y, validation=False, label=iteration + 2, dir = dir)
    model = train_model(model, X, Y, validation=False, label=iteration + 2, dir = dir)
    model = train_model(model, X, Y, validation=False, label=iteration + 2, dir = dir)

    uncertainty = all_branch_uncertainty(model, all_branches, mc=500)
    idx, result = standard_score_of_1478_all(uncertainty, score = score, label=iteration+2, dir = dir)

    # Critic : progress check + new_visiting branches check
    visiting, real_PCT, visiting = critic.record(all_branches, model, X, label=iteration + 2, dir = dir)

    return real_PCT, visiting, result

if __name__ == '__main__':
    algorithm(2, 0)