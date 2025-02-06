from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash_operator import BashOperator
from ultralytics import YOLO
from contextlib import redirect_stdout
import io

def pipeline(model_select, **kwargs):
    model=None
    if model_select=='yolo11n':
        model = YOLO("yolo11n-cls.pt")
    elif model_select=="yolo11s":
        model = YOLO('yolo11s-cls.pt')
    elif model_select=='yolo11m':
        model = YOLO('yolo11m-cls.pt')
    else:
        print("Данная модель не поддерживается")
    e=1
    for k,w in kwargs.items():
        if k=="epoch":
            e=w
    f = io.StringIO()
    with redirect_stdout(f):
        model.train(data='./data/', epochs=e,verbose=False)
        metrics = model.val()
    output = f.getvalue()
    now = datetime.now()
    fn = now.strftime("%Y-%m-%d_%H-%M-%S.txt")
    with open(fn, 'w') as file:
        file.write(output)
    return model, metrics, output


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def prepare_data(**kwargs):
    """Подготовка данных для обучения"""
    print("Подготовка данных...")
    # Здесь ваш код подготовки данных
    return "data_ready"

def train_model(**kwargs):
    """Обучение модели"""
    print("Обучение модели...")
    pipeline('yolo11n',epchos=100)
    # Получаем данные из предыдущего задания
    data_status = kwargs['ti'].xcom_pull(task_ids='prepare_data')
    if data_status != "data_ready":
        raise ValueError("Данные не готовы для обучения")
    # Здесь ваш код обучения модели
    return "model_trained"

def validate_model(**kwargs):
    """Валидация модели"""
    print("Валидация модели...")
    # Получаем статус обучения
    model_status = kwargs['ti'].xcom_pull(task_ids='train_model')
    if model_status != "model_trained":
        raise ValueError("Модель не обучена")
    # Здесь ваш код валидации
    return "model_validated"

def deploy_model(**kwargs):
    """Развертывание модели"""
    print("Развертывание модели...")
    # Получаем статус валидации
    validation_status = kwargs['ti'].xcom_pull(task_ids='validate_model')
    if validation_status != "model_validated":
        raise ValueError("Модель не прошла валидацию")
    # Здесь ваш код развертывания
    return "model_deployed"

with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='DAG для обучения ML модели',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025,2, 4,17),
    catchup=False,
) as dag:
    start = BashOperator(
        task_id='start',
        dag=dag,
    )
    
    prepare_data_task = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
    )
    
    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
    )
    
    end = DummyOperator(
        task_id='end',
        dag=dag,
    )
    