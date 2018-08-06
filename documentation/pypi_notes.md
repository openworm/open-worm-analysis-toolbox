# pypi notes

https://packaging.python.org/tutorials/packaging-projects/

Make sure the following are up to date:
- setuptools
- wheel
- twine

Update the version in version.py ....

Commands:

```python
#From the repo root
python3 setup.py sdist bdist_wheel

#Make sure to remove old versions from dist folder
twine upload dist/*
```
