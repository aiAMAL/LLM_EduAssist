import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

summ_project_name = 'TextSummarization'

list_of_files = [
    '.github/workflows/.gitkeep',
    f'src/{summ_project_name}/__init__.py',
    f'src/{summ_project_name}/entity/__init__.py',
    f'src/{summ_project_name}/entity/entity_config.py',
    f'src/{summ_project_name}/config/__init__.py',
    f'src/{summ_project_name}/config/configuration.py',
    f'src/{summ_project_name}/component/__init__.py',
    f'src/{summ_project_name}/component/data_ingestion.py',
    f'src/{summ_project_name}/component/data_validation.py',
    f'src/{summ_project_name}/component/data_transformation.py',
    f'src/{summ_project_name}/component/model_training.py',
    f'src/{summ_project_name}/component/model_evaluation.py',
    f'src/{summ_project_name}/pipeline/langchain_prediction.py',
    f'src/{summ_project_name}/exception.py',
    f'src/{summ_project_name}/logger.py',
    f'src/{summ_project_name}/utils.py',
    f'templates/index.html',
    f'static/css/styles.css',
    f'FastAPI_html.py',
    'config/config.yaml',
    'config/params.yaml',
    'docker-compose.yaml',
    'Dockerfile',
    'requirements.txt',
    'setup.py',
    'main.py',
    'app.py'
]

for filepath in list_of_files:
    path = Path(filepath)

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f'Created directory at {path.parent} for {path.name}')

    if not path.exists() or path.stat().st_size == 0:
        path.touch()
        logging.info(f'Created empty file {path}')
    else:
        logging.info(f'{path.name} already exists')
