from distutils.core import setup

setup(
    name='PredictPPV',
    version='0.1.0',
    author='M. Michel',
    author_email='guy.inkognito42@gmail.com',
    packages=['predictppv', 'predictppv.parsing', 'predictppv.test'],
    url='https://github.com/MMichel/PredictPPV.git',
    license='LICENSE.txt',
    description='Contact map quality prediction. Predicts positive predictive value of a given amino acid contact map.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.4.1"
        "scipy >= 0.7.2"
        "scikit-learn >= 0.14.1",
    ],
    scripts = ['bin/predict']
)
