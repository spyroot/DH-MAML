from setuptools import setup, find_packages

setup_info = dict(name='DH-MAML',
                  version='1.0',
                  author='Mustafa Bayramov',
                  description='DH-MAML, Distributed Hierarchical Meta Learner',
                  author_email='spyroot@gmail.com',
                  packages=['meta_critics'] + ['meta_critics.' + pkg for pkg in find_packages('meta_critics')],
                  )
setup(**setup_info)
