from threading import Event, Thread

class RenderThread(Thread):
	"""
	Original Code:
		https://github.com/tqjxlm/Simple-DQN-Pytorch/blob/master/Pytorch-DQN-CartPole-Raw-Pixels.ipynb

	Data:
		- Observation: 3 x 400 x 600

	Usage:
		1. call env.step() or env.reset() to update env state
		2. call begin_render() to schedule a rendering task (non-blocking)
		3. call get_screen() to get the lastest scheduled result (block main thread if rendering not done)

	Sample Code:

	```python
		# A simple test
		env = gym.make('CartPole-v0').unwrapped
		renderer = RenderThread(env)
		renderer.start()
		env.reset()
		renderer.begin_render()
		for i in range(100):
			screen = renderer.get_screen() # Render the screen
			env.step(env.action_space.sample()) # Select and perform an action
			renderer.begin_render()
			print(screen)
			print(screen.shape)
		renderer.stop()
		renderer.join()
		env.close()
	```
	"""

	def __init__(self, env):
		super(RenderThread, self).__init__(target=self.render)
		self._stop_event = Event()
		self._state_event = Event()
		self._render_event = Event()
		self.env = env

	def stop(self):
		"""
		Stops the threads

		:return:
		"""
		self._stop_event.set()
		self._state_event.set()

	def stopped(self):
		"""
		Check if the thread has been stopped

		:return:
		"""
		return self._stop_event.is_set()

	def begin_render(self):
		"""
		Start rendering the screen

		:return:
		"""
		self._state_event.set()

	def get_screen(self):
		"""
		get and output the screen image

		:return:
		"""
		self._render_event.wait()
		self._render_event.clear()
		return self.screen

	def render(self):
		while not self.stopped():
			self._state_event.wait()
			self._state_event.clear()
			self.screen = self.env.render(mode='rgb_array')
			self._render_event.set()