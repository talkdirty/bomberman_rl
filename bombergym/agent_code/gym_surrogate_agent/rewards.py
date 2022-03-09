import bombergym.original.events as e

def reward_from_events(events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.COIN_FOUND: 1,
        e.KILLED_SELF: -5,
        e.WAITED: -1,
        e.CRATE_DESTROYED: 1,
        e.MOVED_LEFT: +.1,
        e.MOVED_RIGHT: +.1,
        e.MOVED_UP: +.1,
        e.MOVED_DOWN: +.1,
        e.INVALID_ACTION: -1,
        e.KILLED_OPPONENT: 5,
        e.OPPONENT_ELIMINATED: 5,
        e.SURVIVED_ROUND: 2,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum