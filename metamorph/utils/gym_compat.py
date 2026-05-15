def normalize_reset(reset_result):
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _info = reset_result
        return obs
    return reset_result


def normalize_step(step_result):
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        info = dict(info)
        if truncated:
            info["timeout"] = True
        return obs, reward, done, info
    return step_result
