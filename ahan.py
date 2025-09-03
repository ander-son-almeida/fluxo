from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunNow, SubmitRun, NotebookTask, NewCluster

def executar_notebook_databricks():
    # Inicializa o cliente
    w = WorkspaceClient()
    
    # Método 1: Executar job existente
    run = w.jobs.run_now(
        job_id=123,  # ID do seu job
        notebook_params={"param1": "valor1", "param2": "valor2"}
    )
    
    print(f"Run iniciada: {run.run_id}")
    return run.run_id

def executar_notebook_adhoc():
    w = WorkspaceClient()
    
    # Método 2: Execução ad-hoc
    run = w.jobs.submit(
        run_name="Execução Python Script",
        tasks=[{
            "task_key": "notebook_task",
            "notebook_task": NotebookTask(
                notebook_path="/Users/seu_usuario/seu_notebook",
                base_parameters={"param1": "valor1"}
            ),
            "new_cluster": NewCluster(
                spark_version="11.3.x-scala2.12",
                node_type_id="i3.xlarge",
                num_workers=1
            )
        }]
    )
    
    print(f"Run submetida: {run.run_id}")
    return run.run_id

# Monitorar execução
def monitorar_execucao(run_id):
    w = WorkspaceClient()
    
    while True:
        run_info = w.runs.get(run_id)
        status = run_info.state.life_cycle_state
        
        print(f"Status: {status}")
        
        if status in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
            result_state = run_info.state.result_state
            print(f"Resultado: {result_state}")
            break
            
        time.sleep(10)  # Aguarda 10 segundos

# Exemplo de uso completo
if __name__ == "__main__":
    import time
    
    # Executa o notebook
    run_id = executar_notebook_adhoc()
    
    # Monitora até terminar
    monitorar_execucao(run_id)