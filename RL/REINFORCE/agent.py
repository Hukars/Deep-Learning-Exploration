class Experience:
    def __init__(self, gamma):
        self.transition_list = []
        self.G_t = []
        self.gamma = gamma

    def calculate_return(self):
        next_state_g = 0
        for i in reversed(range(len(self.transition_list))):
            current_transition = self.transition_list[i]
            self.G_t.append(current_transition.reward + self.gamma * next_state_g)
            next_state_g = self.G_t[len(self.G_t)-1]
        self.G_t.reverse()

    def get_item(self, i):
        return self.transition_list[i].state, self.transition_list[i].action, self.G_t[i]

    def reset(self):
        self.transition_list = []
        self.G_t = []
