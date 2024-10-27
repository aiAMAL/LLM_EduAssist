from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

PROJECT_NAME = 'LLM_EduAssist'
VERSION = '0.0.1'
AUTHOR = 'aiAMAL'
REQUIREMENTS_FILE = 'requirements.txt'
DESCRIPTION = 'An AI-powered educational app offering students a GPT chat interface, ' \
              'fine-tuned text summarization, and tools to enhance learning engagement.'


def get_requirements(file_path: str) -> list[str]:
    """
    Reads a requirements file and returns a list of requirements, removing any '-e .' requirement if present.

    :param file_path: The path to the requirements file.
    :return: A list of requirements without the '-e .' requirement.
    """
    with open(file_path, 'r') as file:
        # strips any whitespace (including newlines), & removes empty lines
        requirements = [line.strip() for line in file if line.strip()]

    if HYPHEN_E_DOT in requirements:
        # remove the '-e .' requirement if it exists in the list
        requirements.remove(HYPHEN_E_DOT)

    return requirements


# setup function to define the project details and dependencies
setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    packages=find_packages(),
    install_requires=get_requirements(REQUIREMENTS_FILE),
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url=f'https://github.com/{AUTHOR}/{PROJECT_NAME}',
    python_requires='>=3.9'
)
