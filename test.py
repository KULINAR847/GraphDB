from graphdb import GraphDB
import pickle
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
import json
import contextlib
from copy import deepcopy
import time
import random
import string

# ================== ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ==================

def generate_random_string(length=16):
    """Генерация случайной строки для тестовых данных"""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def test_crud_performance(db: GraphDB, num_operations=1000):
    """Тестирование производительности CRUD операций"""
    results = defaultdict(list)
    
    # Тестирование вставки
    start_time = time.time()
    for i in range(num_operations):
        node_id = f"node_{i}"
        props = {"value": i, "random": generate_random_string()}
        db.insert(node_id, props)
    insert_time = time.time() - start_time
    results["insert"].append(insert_time / num_operations)
    
    #db.print_nodes(300, True)

    # Тестирование чтения
    start_time = time.time()
    for i in range(num_operations):
        db.select(value=i)
    select_time = time.time() - start_time
    results["select"].append(select_time / num_operations)
    
    # Тестирование обновления
    start_time = time.time()
    for i in range(num_operations):
        node_id = f"node_{i}"
        db.update(node_id, {"updated": True, "new_value": i*2})
    update_time = time.time() - start_time
    results["update"].append(update_time / num_operations)
    
    # Тестирование удаления
    start_time = time.time()
    for i in range(num_operations):
        node_id = f"node_{i}"
        db.delete(node_id)
    delete_time = time.time() - start_time
    results["delete"].append(delete_time / num_operations)
    
    return results

def test_relationships_performance(db: GraphDB, num_nodes=100, num_rels_per_node=10):
    """Тестирование производительности операций с отношениями"""
    results = {}
    
    # Создаем узлы
    for i in range(num_nodes):
        db.insert(f"node_{i}", {"type": "test", "index": i})
    
    # Тестирование добавления отношений
    start_time = time.time()
    for i in range(num_nodes):
        for j in range(num_rels_per_node):
            target = random.randint(0, num_nodes-1)
            db.add_relationship(
                f"node_{i}", 
                f"node_{target}", 
                "RELATES_TO",
                {"weight": random.random()}
            )
    results["add_relationship"] = time.time() - start_time
    
    # Тестирование поиска отношений
    start_time = time.time()
    for i in range(num_nodes):
        db.find_relationships(from_node=f"node_{i}")
    results["find_relationships"] = time.time() - start_time
    
    # Очистка
    for i in range(num_nodes):
        db.delete(f"node_{i}")
    
    return results

def test_index_performance(db: GraphDB, num_nodes=1000, num_queries=100):
    """Тестирование эффективности индексов"""
    results = {}
    
    # Создаем узлы с различными свойствами
    for i in range(num_nodes):
        db.insert(f"node_{i}", {
            "age": random.randint(18, 80),
            "category": random.choice(["A", "B", "C", "D"]),
            "active": random.choice([True, False])
        })
    
    # Тестирование поиска без индексов (по всем узлам)
    start_time = time.time()
    for _ in range(num_queries):
        age = random.randint(18, 80)
        db.select(age=age)
    time_without_index = time.time() - start_time
    results["select_without_index"] = time_without_index if time_without_index > 0 else 0.0001
    
    # Создаем индексы
    db._indexes = {}  # Сброс индексов
    for i in range(num_nodes):
        db._update_indexes(f"node_{i}", db.nodes[f"node_{i}"]["props"])
    
    # Тестирование поиска с индексами
    start_time = time.time()
    for _ in range(num_queries):
        age = random.randint(18, 80)
        db.select(age=age)
    time_with_index = time.time() - start_time
    results["select_with_index"] = time_with_index if time_with_index > 0 else 0.0001
    
    # Очистка
    for i in range(num_nodes):
        db.delete(f"node_{i}")
    
    return results

def test_transaction_performance(db: GraphDB, num_transactions=100, ops_per_transaction=10):
    """Тестирование производительности транзакций"""
    results = []
    
    for _ in range(num_transactions):
        start_time = time.time()
        with db.transaction():
            for i in range(ops_per_transaction):
                node_id = f"trans_node_{random.randint(0, 1000000)}"
                if random.random() > 0.5:
                    db.insert(node_id, {"value": random.randint(0, 100)})
                elif node_id in db.nodes:
                    if random.random() > 0.5:
                        db.update(node_id, {"updated": True})
                    else:
                        db.delete(node_id)
        results.append(time.time() - start_time)
    
    return {
        "avg_time": sum(results) / len(results),
        "min_time": min(results),
        "max_time": max(results)
    }

def test_scalability(db: GraphDB, max_nodes=200000, step=10000):
    """Тестирование масштабируемости при увеличении количества данных"""
    print("\nТестирование масштабируемости:")
    print("Количество узлов | Время вставки | Время поиска")
    
    results = []
    
    for n in range(step, max_nodes + 1, step):
        # Очистка базы
        db.nodes.clear()
        db.relationships.clear()
        db._indexes.clear()
        
        # Тестирование вставки
        start_time = time.time()
        for i in range(n):
            db.insert(f"node_{i}", {"value": i, "group": i % 10})
        insert_time = time.time() - start_time
        
        # Тестирование поиска
        start_time = time.time()
        for _ in range(100):
            group = random.randint(0, 9)
            db.select(group=group)
        select_time = time.time() - start_time
        
        results.append((n, insert_time, select_time))
        print(f"{n:14} | {insert_time:12.4f} | {select_time:11.4f}")
    
    return results

def get_time():
    return time.time()

def run_comprehensive_load_test(db: GraphDB):
    """Комплексное нагрузочное тестирование"""
    print("=== Начало комплексного нагрузочного тестирования ===")
    
    # Тестирование CRUD операций с разным количеством записей
    print("\n1. Тестирование CRUD операций:")
    for size in [100, 1000, 100000]:  # Уменьшил максимальный размер для теста
        print(f"\nРазмер данных: {size} записей")
        start_time = get_time()       
        crud_results = test_crud_performance(db, size)
        test_crud_performance_time = get_time() - start_time
        
        for op, times in crud_results.items():
            print(f"{op}: {sum(times)/len(times):.6f} сек/операция")
        print(f'test_crud_performance_time = {test_crud_performance_time}')
    
    # Тестирование отношений
    print("\n2. Тестирование работы с отношениями:")
    rel_results = test_relationships_performance(db, 100, 10)
    for op, time in rel_results.items():
        print(f"{op}: {time:.4f} сек")
    
    # Тестирование индексов
    print("\n3. Тестирование эффективности индексов:")
    index_results = test_index_performance(db, 1000, 100)
    print(f"Поиск без индекса: {index_results['select_without_index']:.4f} сек")
    print(f"Поиск с индексом: {index_results['select_with_index']:.4f} сек")
    
    # Добавляем проверку деления на ноль
    if index_results['select_without_index'] > 0:
        improvement = ((index_results['select_without_index'] - index_results['select_with_index']) / 
                      index_results['select_without_index']) * 100
        print(f"Улучшение производительности: {improvement:.2f}%")
    else:
        print("Улучшение производительности: N/A (время поиска без индекса равно 0)")
    
    # Тестирование транзакций
    print("\n4. Тестирование транзакций:")
    trans_results = test_transaction_performance(db, 100, 10)
    print(f"Среднее время транзакции: {trans_results['avg_time']:.4f} сек")
    print(f"Минимальное время: {trans_results['min_time']:.4f} сек")
    print(f"Максимальное время: {trans_results['max_time']:.4f} сек")
    
    print("\n=== Тестирование завершено ===")

if __name__ == "__main__":
    # Инициализация базы данных
    db = GraphDB("test_db.pickle")
    
    # Запуск комплексного тестирования
    run_comprehensive_load_test(db)
    
    # Дополнительное тестирование масштабируемости
    scalability_results = test_scalability(db)