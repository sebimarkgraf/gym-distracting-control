import gym


def test_max_episode_steps():
    env = gym.make('distracting_control/Walker-walk-easy-v1')
    steps = 0
    done = False
    env.reset()
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        steps += 1

    assert steps == 250



def test_flat_obs():
    env = gym.make('distracting_control/Walker-walk-easy-v1', frame_skip=4)
    assert env.reset().shape == (24,)


def test_frame_skip():
    env = gym.make('distracting_control/Walker-walk-easy-v1', from_pixels=True, frame_skip=8)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        steps += 1

    assert steps == 125
    # default channel goes first
    assert env.reset().shape == (3, 84, 84)


def test_channel_first():
    env = gym.make('distracting_control/Walker-walk-easy-v1', from_pixels=True, channels_first=True)
    assert env.reset().shape == (3, 84, 84)


def test_channel_last():
    env = gym.make('distracting_control/Walker-walk-easy-v1', from_pixels=True, frame_skip=8, channels_first=False)
    assert env._max_episode_steps == 125
    assert env.reset().shape == (84, 84, 3)


def test_gray_scale():
    """
    this currently does not support gray scale, because the DMControl rendering function
    only supports RGB, segmentation mask and so on.

    :return:
    """
    pass
