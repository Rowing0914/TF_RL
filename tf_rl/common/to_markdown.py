# reference: https://github.com/thombashi/pytablewriter
from pytablewriter import MarkdownTableWriter


def params_to_markdown(params_str):
    """ this converts the hyper-params into MarkDown text and we put it on Tensorboard subsequently """

    params = process_text(text=params_str)

    writer = MarkdownTableWriter()
    writer.table_name = "Hyper-parameters"
    writer.headers = ["Item", "Value"]

    items = []
    for key, value in params.items():
        items.append([str(key), str(value)])

    writer.value_matrix = items
    writer.margin = 1  # add a whitespace for both sides of each cell

    return writer.dumps()


def process_text(text):
    # at this point, list_str contains bunch of 'batch_size = 32 '-ish params inside the list
    list_str = "".join(text.split("\n")[4:-1]).split("train_eval.")[1:]  # TODO: this is really bad practice
    params = dict()

    for _str in list_str:
        _str.replace(" ", "")
        param = _str.split("=")
        params[param[0]] = param[1]

    return params
