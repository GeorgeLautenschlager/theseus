from typing import List


class Autocore:
  def __init__(self):
    self.tasks: List[str] = []
    self.current_task_uuid: str = ""
    self.goals: List[str] = []
    self.telos: str = ""
