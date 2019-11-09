"""
Test to see if we can override the params defined in gin_file upon execution.

```shell
python tf_rl/test/utils_test/gin_test/gin_config.py --gin_params=mock.a=0
```

"""

import gin
import argparse


@gin.configurable
def mock(a, b, c, d, e):
    print(a, b, c, d, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_file", default="test.gin", help="test gin file")
    parser.add_argument("--gin_params", action='append', nargs='+', help="extra gin override params")
    params = parser.parse_args()
    print(params.gin_params)
    a = [s[0] for s in params.gin_params]
    print(a)
    gin.parse_config_file(params.gin_file)
    gin.parse_config_files_and_bindings([params.gin_file], a)
    mock()
