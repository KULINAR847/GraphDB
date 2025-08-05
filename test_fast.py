import time
import random
from collections import defaultdict
from graphdb import GraphDB
import string

def generate_random_string(length=16):
    """Генерация случайной строки для тестовых данных"""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def test_extended_crud_performance(db: GraphDB, num_operations=1000):
    """Расширенное тестирование CRUD операций с измерением каждого типа операций"""
    results = {
        'insert': [],
        'select': [],
        'update': [],
        'delete': []
    }
    
    # Тестирование вставки
    for i in range(num_operations):
        node_id = f"node_{i}"
        props = {"value": i, "random": generate_random_string()}
        start = time.perf_counter()
        db.insert(node_id, props)
        results['insert'].append(time.perf_counter() - start)
    
    # Тестирование чтения
    for i in range(num_operations):
        node_id = f"node_{i}"
        start = time.perf_counter()
        db.select(value=i)
        results['select'].append(time.perf_counter() - start)
    
    # Тестирование обновления
    for i in range(num_operations):
        node_id = f"node_{i}"
        start = time.perf_counter()
        db.update(node_id, {"updated": True, "new_value": i*2})
        results['update'].append(time.perf_counter() - start)
    
    # Тестирование удаления
    for i in range(num_operations):
        node_id = f"node_{i}"
        start = time.perf_counter()
        db.delete(node_id)
        results['delete'].append(time.perf_counter() - start)
    
    return results

def test_memory_usage(db: GraphDB, num_nodes=10000):
    """Тестирование использования памяти"""
    import sys
    from memory_profiler import memory_usage
    
    # Очистка базы
    db.nodes.clear()
    db.relationships.clear()
    db._indexes.clear()
    
    # Измерение памяти перед тестом
    mem_before = memory_usage(-1, interval=0.1)[0]
    
    # Вставка узлов
    for i in range(num_nodes):
        db.insert(f"node_{i}", {
            "id": i,
            "data": generate_random_string(100),
            "group": i % 10
        })
    
    # Измерение памяти после вставки
    mem_after = memory_usage(-1, interval=0.1)[0]
    
    # Расчет использования памяти на узел
    mem_per_node = (mem_after - mem_before) / num_nodes
    
    # Очистка
    db.nodes.clear()
    db.relationships.clear()
    db._indexes.clear()
    
    return {
        "memory_used_mb": mem_after - mem_before,
        "memory_per_node_kb": mem_per_node * 1024
    }

def test_concurrency(db: GraphDB, num_threads=4, ops_per_thread=250):
    """Тестирование конкурентного доступа"""
    from threading import Thread
    
    results = []
    
    def worker(thread_id):
        start = time.time()
        for i in range(ops_per_thread):
            node_id = f"thread_{thread_id}_node_{i}"
            db.insert(node_id, {"thread": thread_id, "value": i})
            db.select(thread=thread_id)
            db.update(node_id, {"updated": True})
            db.delete(node_id)
        results.append(time.time() - start)
    
    # Очистка базы перед тестом
    db.nodes.clear()
    db.relationships.clear()
    db._indexes.clear()
    
    threads = []
    for i in range(num_threads):
        t = Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total_ops = num_threads * ops_per_thread * 4  # 4 операции на запрос
    total_time = max(results)
    
    return {
        "ops_per_sec": total_ops / total_time,
        "total_time": total_time,
        "thread_times": results
    }

def run_advanced_load_test(db: GraphDB):
    """Расширенное нагрузочное тестирование"""
    print("=== Расширенное нагрузочное тестирование ===")
    
    # 1. Расширенное тестирование CRUD
    print("\n1. Расширенное тестирование CRUD (1000 операций):")
    crud_results = test_extended_crud_performance(db, 1000)
    for op, times in crud_results.items():
        avg = sum(times) / len(times)
        print(f"{op}: avg={avg:.6f}s, min={min(times):.6f}s, max={max(times):.6f}s")
    
    # 2. Тестирование использования памяти
    print("\n2. Тестирование использования памяти (1,000,000 узлов):")
    mem_results = test_memory_usage(db, 1000000)
    print(f"Использовано памяти: {mem_results['memory_used_mb']:.2f} MB")
    print(f"Памяти на узел: {mem_results['memory_per_node_kb']:.2f} KB")
    
    # 3. Тестирование конкурентности
    print("\n3. Тестирование конкурентного доступа (4 потока):")
    conc_results = test_concurrency(db, 4, 250)
    print(f"Операций в секунду: {conc_results['ops_per_sec']:.2f}")
    print(f"Общее время: {conc_results['total_time']:.2f}s")
    print(f"Время по потокам: {conc_results['thread_times']}")
    
    # 4. Детальное тестирование масштабируемости
    print("\n4. Детальное тестирование масштабируемости:")
    print("Узлы | Вставка | Поиск | Обновление | Удаление")
    for size in [10000, 20000, 500000, 1000000]:
        # Очистка
        db.nodes.clear()
        db.relationships.clear()
        db._indexes.clear()
        
        # Тест вставки
        start = time.time()
        for i in range(size):
            db.insert(f"node_{i}", {"value": i, "group": i % 10})
        insert_time = time.time() - start
        
        # Тест поиска
        start = time.time()
        for _ in range(100):
            group = random.randint(0, 9)
            db.select(group=group)
        select_time = time.time() - start
        
        # Тест обновления
        start = time.time()
        for i in range(100):
            db.update(f"node_{i}", {"updated": True})
        update_time = time.time() - start
        
        # Тест удаления
        start = time.time()
        for i in range(100):
            if f"node_{i}" in db.nodes:
                db.delete(f"node_{i}")
        delete_time = time.time() - start
        
        print(f"{size:6} | {insert_time:7.4f} | {select_time:6.4f} | {update_time:8.4f} | {delete_time:7.4f}")
    
    print("\n=== Тестирование завершено ===")

if __name__ == "__main__":
    db = GraphDB("advanced_test_db.pickle")
    
    # Очистка базы перед тестами
    db.nodes.clear()
    db.relationships.clear()
    db._indexes.clear()
    
    # Запуск расширенных тестов
    run_advanced_load_test(db)
    
    # Дополнительные тесты
    print("\nДополнительные тесты:")
    
    # Тестирование сложных запросов
    print("\nТестирование сложных запросов:")
    db.nodes.clear()
    for i in range(1000):
        db.insert(f"user_{i}", {
            "age": random.randint(18, 65),
            "salary": random.randint(30000, 150000),
            "department": random.choice(["IT", "HR", "Finance", "Sales"]),
            "active": random.choice([True, False])
        })
    
    start = time.time()
    results = db.select(age__gt=30, salary__lt=100000, department="IT", active=True)
    print(f"Сложный запрос: {time.time() - start:.4f}s, найдено: {len(results)}")
    
    # Тестирование транзакций с откатом
    print("\nТестирование транзакций с откатом:")
    try:
        with db.transaction():
            db.insert("temp_1", {"test": True})
            db.insert("temp_2", {"test": True})
            raise Exception("Имитация ошибки")
    except:
        print("Транзакция откатилась, temp_1 существует:", "temp_1" in db.nodes)