import bombergym.original.events as e

def reward_from_events(events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -1,
        e.KILLED_OPPONENT: 10,
        e.SURVIVED_ROUND: 10,
        e.KILLED_SELF: -10,
        e.MOVED_DOWN: -.5,
        e.MOVED_LEFT: -.5,
        e.MOVED_RIGHT: -.5,
        e.MOVED_UP: -.5,
        e.WAITED: -.8,
        e.CRATE_DESTROYED: 3,
        e.BOMB_DROPPED: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
