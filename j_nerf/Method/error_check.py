import jittor as jt


def checkError(data, data_name, is_exit=True):
    if jt.__version__.split(".")[2] != 6:
        return
    is_inf = jt.isinf(data)
    if True in is_inf:
        print("[ERROR][check_error::checkError]")
        print(
            "\t INF found in " + data_name + "! sum:",
            jt.where(is_inf)[0].shape,
            "/",
            data.shape,
        )
        if is_exit:
            exit()
    is_nan = jt.isnan(data)
    if True in is_nan:
        print("[ERROR][check_error::checkError]")
        print(
            "\t nan found in " + data_name + "! sum:",
            jt.where(is_nan)[0].shape,
            "/",
            data.shape,
        )
        if is_exit:
            exit()
    return True
