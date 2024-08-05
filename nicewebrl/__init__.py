
import os.path


def basic_javascript_file():
  current_file_path = os.path.abspath(__file__)
  current_directory = os.path.dirname(current_file_path)
  file = f"{current_directory}/basics.js"
  return file
