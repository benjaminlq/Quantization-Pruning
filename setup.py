from setuptools import setup

if __name__ == "__main__":
    setup()

# def get_install_requirements():
#     with open("requirements.txt", "r", encoding="utf-8") as f:
#         reqs = [x.strip() for x in f.read().splitlines()]
#     reqs = [x for x in reqs if not x.startswith("#")]
#     return reqs

# setup(
#     install_requires=get_install_requirements(),
# )