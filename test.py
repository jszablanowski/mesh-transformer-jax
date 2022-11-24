def get_file_size():
    with open("test.py", "r") as txt_file:
        context = txt_file.read()
    return len(context)