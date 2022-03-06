from stable_baselines3.common.callbacks import BaseCallback
import pickle


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(
        self, 
        save_directory,
        every_n_timesteps=1024, 
        num_samples=5,
        save_checkpoints=True,
        verbose=0,
        ):
        super(CustomCallback, self).__init__(verbose)
        self.every_n_timesteps = every_n_timesteps
        self.num_samples = num_samples
        self.save_checkpoints = save_checkpoints
        self.save_directory = save_directory
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def checkpoint(self):
        filename = f"{self.save_directory}/model.{self.n_calls}"
        self.model.save(filename)
        self.logger.info(f"Successful checkpoint of model to {filename}")

    def sample_model(self):
        episode_recordings = []
        total_episodes = 0
        total_reward = 0
        for _ in range(self.num_samples):
            episode_buffer = []
            obs = self.training_env.reset()
            episode_length = 0
            episode_reward = 0
            while True:
                action, _ = self.model.predict(obs)
                obs, rew, done, other = self.training_env.step(action)
                episode_length += 1
                episode_reward += rew[0]
                # Only save first episode in batch
                episode_buffer.append((obs, action[0], rew[0], done[0], other[0]))
                if done[0]: 
                    break
            episode_recordings.append(episode_buffer)
            total_reward += episode_reward
            total_episodes += episode_length
        filename = f"{self.save_directory}/recordings.{self.n_calls}.pcl"
        with open(filename, 'wb') as f:
            self.logger.info(f"Successfully saved {filename}. Avg. Episode Length: {total_episodes/self.num_samples}. Avg Reward: {total_reward/self.num_samples}")
            pickle.dump(episode_recordings, f)
       

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % self.every_n_timesteps == 0:
            if self.save_checkpoints:
                self.checkpoint()
            self.sample_model()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass