
### Use this space to try out ideas and free code ###
import json

contents = { 'error': 1, 'Object_2': "fsdfs", 'prop_3': '323dd' }

with open('load.json', 'r+') as f:
  file_obj = json.load(f)
  print(file_obj)
  file_obj['errors'].append(contents)
  f.seek(0)
  json.dump(file_obj, f)
