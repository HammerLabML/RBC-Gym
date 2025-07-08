import gymnasium as gym


class RBCNormalizeReward(gym.RewardWrapper):
    def __init__(self, env, ra, s, a):
        r"""
            The maximum Nusselt number depends on the Rayleigh number with a power law: Nu ~ sRa^a
    
            Default values:  
            3D: s=0.22, a=0.27  
            2D: s=0.1 , a=0.4
        """
        super().__init__(env)
        self.scale = s * (ra**a)

    def reward(self, reward):
        return (reward + self.scale) / self.scale
