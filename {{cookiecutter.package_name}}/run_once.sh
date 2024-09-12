github_url="{{cookiecutter.package_url}"
git init
git remote add origin github_url

poetry add pre-commit
pre-commit install

poetry add black isort flake8 --group formatting
poetry add mypy --group typechecking
poetry add pytest pytest-cov --group testing

poetry add torch torchvision pandas matplotlib albumentations scikit-learn tqdm
poetry add sphinx insegal --group docs
