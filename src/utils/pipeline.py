# -----------------------------------------------------------------------------
# Submitted by:     u3258928
# Date:             2024 November 02
# Course:           Software Technology 1
# Activity:         Assessment 3: Programming project
# -----------------------------------------------------------------------------


class Pipeline:
    def __init__(self):
        self.steps = []
        self.context = {}

    def add_step(self, func):
        self.steps.append(func)

    def execute(self):
        for step in self.steps:
            self.context = step(self.context)
        return self.context
