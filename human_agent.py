from enum import Enum

class Action(Enum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass= 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18
    NotUsed = 55


sticky_index_to_action = [
    Action.Left,
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.Right,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Sprint,
    Action.Dribble
]


class PlayerRole(Enum):
    GoalKeeper = 0
    CenterBack = 1
    LeftBack = 2
    RightBack = 3
    DefenceMidfield = 4
    CentralMidfield = 5
    LeftMidfield = 6
    RIghtMidfield = 7
    AttackMidfield = 8
    CentralFront = 9


class GameMode(Enum):
    Normal = 0
    KickOff = 1
    GoalKick = 2
    FreeKick = 3
    Corner = 4
    ThrowIn = 5
    Penalty = 6

def simple_human_policy(obs):
    # Make sure player is running.
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    # We always control left team (observations and actions
    # are mirrored appropriately by the environment).
    controlled_player_pos = obs['left_team'][obs['active']]
    # Does the player we control have the ball?
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        # Shot if we are 'close' to the goal (based on 'x' coordinate).
        if controlled_player_pos[0] > 0.5:
            return Action.Shot
        # Run towards the goal otherwise.
        return Action.Right
    else:
        # Run towards the ball.
        if obs['ball'][0] > controlled_player_pos[0] + 0.05:
            return Action.Right
        if obs['ball'][0] < controlled_player_pos[0] - 0.05:
            return Action.Left
        if obs['ball'][1] > controlled_player_pos[1] + 0.05:
            return Action.Bottom
        if obs['ball'][1] < controlled_player_pos[1] - 0.05:
            return Action.Top
        # Try to take over the ball if close to the ball.
        return Action.Slide
    

def human_agent_wrapper(obs):
    # Extract observations for the first (and only) player we control.
    obs_hr = obs['players_raw'][0]
    # Turn 'sticky_actions' into a set of active actions (strongly typed).
    obs_hr['sticky_actions'] = { sticky_index_to_action[nr] for nr, action in enumerate(obs_hr['sticky_actions']) if action }
    # Turn 'game_mode' into an enum.
    obs_hr['game_mode'] = GameMode(obs_hr['game_mode'])
    return obs_hr


def agent(obs):

    a =  simple_human_policy(human_agent_wrapper(obs)).value
    
    return [a]

