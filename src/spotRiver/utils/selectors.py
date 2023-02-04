from river.tree.splitter import EBSTSplitter, QOSplitter


def select_splitter(i):
    if i == 0:
        return EBSTSplitter()
    else:
        return QOSplitter()
