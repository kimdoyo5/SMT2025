import pandas as pd
from sportypy.surfaces import MiLBField
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import HTML

def plot_animation(player_position_df: pd.DataFrame, 
                   ball_position_df: pd.DataFrame, 
                   play_id: int = 1, 
                   save_gif: bool = False) -> HTML:
    
    """
    A function that plots field animations for a particular instance of a game.
    Example usage:
    
    ```
    player_pos = pd.read_parquet('./Combined/player_pos/1884_110_Vis2AR_Home2A.parquet')
    ball_pos = pd.read_parquet('./Combined/ball_pos/1884_110_Vis2AR_Home2A.parquet')
    
    plot_animation(player_pos, ball_pos, play_id = 30, save_gif = True)
    ```
    
    Params:
        player_position_df: A Data Frame for the player position coordinates on a field.
        ball_position_df: A Data Frame for the ball position coordiantes on a field.
        play_id: A integer field that denotes the play id you want to visualize. Defaults to 1.
        save_gif: A boolean that when set to true saves the animation to a gif. 
                  Defaults to not saving animation (False).
    
    Returns:
        HTML display of the animation. You can slow or speed up the fps with the + and - buttons.
        You can also press the > arrow to start the animation.
    """
    
    if not isinstance(play_id, int):
        raise ValueError("Play ID must be an Integer. This function only handles one Play ID.")
    
    if len(player_position_df['game_str'].unique()) > 1 or len(ball_position_df['game_str'].unique()) > 1:
        raise ValueError("Player Position or Ball Position Data Frame has multiple games. Please filter for one game at a time.")
    
    player_pos = player_position_df.query(f'play_id == {play_id}')
    ball_pos = ball_position_df.query(f'play_id == {play_id}')
    
    merged_df = pd.merge(player_pos, ball_pos, on = ['timestamp', 'play_id', 'game_str'], how = 'left')
    merged_df = merged_df[merged_df['player_position'] < 14] # Elminate umpires and coaches on field
    
    field = MiLBField()
    field.draw(display_range='full')

    fig = plt.gcf()
    ax = plt.gca()

    p = field.scatter([], [], c='white')
    b = field.scatter([], [], c='red')

    game_id = merged_df['game_str'].unique()[0]
    game_text = ax.text(0, 400, f'Game ID: {game_id}', c='white', ha='center')
    play_text = ax.text(120, 0, f'Play: {play_id}', c='white', ha='center')


    def update(frame):
        frame_data = merged_df[merged_df['timestamp'] <= frame]

        players = frame_data.sort_values('timestamp').drop_duplicates(subset=['player_position'], keep='last')
        balls = frame_data[['ball_position_x', 'ball_position_y', 'ball_position_z']].dropna().iloc[-1:]

        players_colors = ['yellow' if 10 <= pos <= 13 else 'white' for pos in players['player_position']]

        p.set_offsets(np.c_[players['field_x'], players['field_y']])
        p.set_color(players_colors)

        ball_size = (balls['ball_position_z'].values * 8)
        b.set_offsets(np.c_[balls['ball_position_x'], balls['ball_position_y']])
        
        if ball_size < 1:
             ball_size = np.array([10])
                

        return p, b

    ani = FuncAnimation(fig, update, frames=np.linspace(merged_df['timestamp'].min(), 
                                                        merged_df['timestamp'].max(), num=50), blit=True)

    if save_gif:
        ani.save('animation.gif', writer='imagemagick', fps=10)
    
    return HTML(ani.to_jshtml())
