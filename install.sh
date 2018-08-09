cd src/main/python
python ../../../setup.py build bdist_egg
easy_install dist/aihubcore-*.egg
rm -rf aihubcore.egg-info
rm -rf build
rm -rf dist
cd ../../..
