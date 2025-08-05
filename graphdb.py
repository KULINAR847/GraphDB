import pickle
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from collections import defaultdict
import threading
from dataclasses import dataclass
from copy import deepcopy
import zlib
import random
import contextlib
import numpy as np

# Для многопоточности
db_lock = threading.RLock()

@dataclass
class Relationship:
    from_node: str
    to_node: str
    type: str
    props: Dict[str, Any]

class GraphDB:
    def __init__(self, filepath: str = "graph_db.pickle", use_compression: bool = False):
        self.filepath = filepath
        self.use_compression = use_compression
        
        # Основные структуры данных
        self.nodes: Dict[str, Dict] = {}
        self.relationships: Dict[str, List[Relationship]] = defaultdict(list)
        self._indexes: Dict[str, Dict[Any, Set[str]]] = {}
        
        # Кэши
        self._select_cache: Dict[Tuple, List[Dict]] = {}
        self._rel_cache: Dict[Tuple, List[Relationship]] = {}
        
        # Для алгоритмов анализа графа
        self._shortest_paths: Optional[np.ndarray] = None
        self._node_ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        
        # Статистика
        self.stats = {
            'operations': defaultdict(int),
            'performance': defaultdict(list)
        }

    # ================== АНАЛИЗ ГРАФА ==================
    
    def precompute_shortest_paths(self):
        """Предварительное вычисление кратчайших путей (Флойд-Уоршелл)"""
        with db_lock:
            node_ids = list(self.nodes.keys())
            size = len(node_ids)
            
            if size == 0:
                return
                
            # Создаем mapping ID -> индекс
            self._node_ids = node_ids
            self._id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
            
            # Инициализация матрицы расстояний
            dist = np.full((size, size), np.inf)
            np.fill_diagonal(dist, 0)
            
            # Заполняем матрицу на основе связей
            for from_node, rels in self.relationships.items():
                i = self._id_to_idx[from_node]
                for rel in rels:
                    j = self._id_to_idx[rel.to_node]
                    dist[i][j] = 1  # Все связи имеют вес 1
            
            # Алгоритм Флойда-Уоршелла
            for k in range(size):
                for i in range(size):
                    for j in range(size):
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
            
            self._shortest_paths = dist

    def get_shortest_path(self, from_node: str, to_node: str) -> float:
        """Получение предвычисленного кратчайшего пути"""
        with db_lock:
            if self._shortest_paths is None:
                self.precompute_shortest_paths()
                
            if from_node not in self._id_to_idx or to_node not in self._id_to_idx:
                return float('inf')
                
            i = self._id_to_idx[from_node]
            j = self._id_to_idx[to_node]
            return self._shortest_paths[i][j]

    # ================== ОСНОВНЫЕ МЕТОДЫ ==================
    
    def insert(self, node_id: str, props: Optional[Dict] = None):
        """Добавление узла"""
        with db_lock:
            if props is None:
                props = {}
            self.nodes[node_id] = {"props": props, "edges": []}
            self._update_indexes(node_id, props)
            self._clear_caches()
            self._invalidate_graph_analysis()
            self.stats['operations']['insert'] += 1

    def bulk_insert(self, nodes_data: Dict[str, Dict]):
        """Массовая вставка узлов"""
        with db_lock:
            for node_id, props in nodes_data.items():
                self.nodes[node_id] = {"props": props, "edges": []}
                self._update_indexes(node_id, props)
            self._clear_caches()
            self._invalidate_graph_analysis()
            self.stats['operations']['bulk_insert'] += 1

    def update(self, node_id: str, new_props: Dict):
        """Обновление узла"""
        with db_lock:
            if node_id not in self.nodes:
                raise ValueError(f"Узел {node_id} не найден!")
            
            old_props = self.nodes[node_id]["props"]
            self.nodes[node_id]["props"].update(new_props)
            self._remove_from_indexes(node_id, old_props)
            self._update_indexes(node_id, new_props)
            self._clear_caches()
            self._invalidate_graph_analysis()
            self.stats['operations']['update'] += 1

    def delete(self, node_id: str):
        """Удаление узла"""
        with db_lock:
            if node_id not in self.nodes:
                raise ValueError(f"Узел {node_id} не найден!")
            
            # Удаляем связи
            if node_id in self.relationships:
                del self.relationships[node_id]
            
            # Удаляем ссылки на узел в других связях
            for rels in self.relationships.values():
                rels[:] = [rel for rel in rels if rel.to_node != node_id]
            
            # Удаляем из индексов
            self._remove_from_indexes(node_id, self.nodes[node_id]["props"])
            del self.nodes[node_id]
            self._clear_caches()
            self._invalidate_graph_analysis()
            self.stats['operations']['delete'] += 1

    def _invalidate_graph_analysis(self):
        """Сброс предвычисленных данных графа"""
        self._shortest_paths = None
        self._node_ids = []
        self._id_to_idx = {}

    # ================== ПОИСК И ФИЛЬТРАЦИЯ ==================
    
    def select(self, **filters) -> List[Dict]:
        """Поиск узлов с кэшированием"""
        cache_key = tuple(sorted(filters.items()))
        
        if cache_key in self._select_cache:
            return self._select_cache[cache_key]
            
        with db_lock:
            if not filters:
                result = [{"id": n, **d["props"]} for n, d in self.nodes.items()]
                self._select_cache[cache_key] = result
                return result
            
            # Поиск с использованием индексов
            result_ids = None
            for key, value in filters.items():
                if key in self._indexes and value in self._indexes[key]:
                    matched = self._indexes[key][value]
                    result_ids = matched if result_ids is None else result_ids & matched
                    if not result_ids:
                        self._select_cache[cache_key] = []
                        return []
            
            if result_ids is not None:
                result = [{"id": n, **self.nodes[n]["props"]} for n in result_ids]
                self._select_cache[cache_key] = result
                return result
            
            # Полный перебор
            result = [{"id": n, **d["props"]} for n, d in self.nodes.items() 
                     if all(d["props"].get(k) == v for k, v in filters.items())]
            self._select_cache[cache_key] = result
            return result

    # ================== РАБОТА С СВЯЗЯМИ ==================
    
    def add_relationship(self, from_node: str, to_node: str, rel_type: str, props: Optional[Dict] = None):
        """Добавление связи"""
        with db_lock:
            if props is None:
                props = {}
            if from_node not in self.nodes or to_node not in self.nodes:
                raise ValueError("Один из узлов не найден!")
            
            rel = Relationship(from_node, to_node, rel_type, props)
            self.relationships[from_node].append(rel)
            self.nodes[from_node]["edges"].append((to_node, rel_type))
            self._clear_rel_cache()
            self._invalidate_graph_analysis()
            self.stats['operations']['add_relationship'] += 1

    def bulk_add_relationships(self, relationships: List[Dict]):
        """Массовое добавление связей"""
        with db_lock:
            for rel in relationships:
                self.add_relationship(**rel)

    def find_relationships(self, from_node: Optional[str] = None, 
                         to_node: Optional[str] = None, 
                         rel_type: Optional[str] = None) -> List[Relationship]:
        """Поиск связей с кэшированием"""
        cache_key = (from_node, to_node, rel_type)
        
        if cache_key in self._rel_cache:
            return self._rel_cache[cache_key]
            
        with db_lock:
            if from_node is not None:
                rels = self.relationships.get(from_node, [])
                result = [rel for rel in rels
                         if (to_node is None or rel.to_node == to_node) and
                         (rel_type is None or rel.type == rel_type)]
            else:
                result = [rel for rels in self.relationships.values() 
                         for rel in rels
                         if (to_node is None or rel.to_node == to_node) and
                         (rel_type is None or rel.type == rel_type)]
            
            self._rel_cache[cache_key] = result
            return result

    # ================== УПРАВЛЕНИЕ КЭШЕМ ==================
    
    def _clear_caches(self):
        """Очистка всех кэшей"""
        self._select_cache.clear()
        self._rel_cache.clear()

    def _clear_rel_cache(self):
        """Очистка кэша связей"""
        self._rel_cache.clear()

    # ================== ИНДЕКСАЦИЯ ==================
    
    def _update_indexes(self, node_id: str, props: Dict):
        """Обновление индексов"""
        for key, value in props.items():
            if key not in self._indexes:
                self._indexes[key] = defaultdict(set)
            self._indexes[key][value].add(node_id)

    def _remove_from_indexes(self, node_id: str, props: Dict):
        """Удаление из индексов"""
        for key, value in props.items():
            if key in self._indexes and value in self._indexes[key] and node_id in self._indexes[key][value]:
                self._indexes[key][value].remove(node_id)
                if not self._indexes[key][value]:
                    del self._indexes[key][value]

    def create_index(self, property_name: str):
        """Создание индекса"""
        with db_lock:
            if property_name not in self._indexes:
                self._indexes[property_name] = defaultdict(set)
                for node_id, data in self.nodes.items():
                    if property_name in data["props"]:
                        self._indexes[property_name][data["props"][property_name]].add(node_id)
                self._clear_caches()

    # ================== СЕРИАЛИЗАЦИЯ ==================
    
    def save(self):
        """Сохранение данных"""
        with db_lock:
            data = {
                "nodes": self.nodes,
                "relationships": {k: [rel.__dict__ for rel in v] for k, v in self.relationships.items()},
                "_indexes": {k: {vk: list(vv) for vk, vv in v.items()} for k, v in self._indexes.items()}
            }
            
            serialized = pickle.dumps(data)
            if self.use_compression:
                serialized = zlib.compress(serialized)
            
            with open(self.filepath, 'wb') as f:
                f.write(serialized)

    def load(self):
        """Загрузка данных"""
        try:
            with open(self.filepath, 'rb') as f:
                serialized = f.read()
                
            if self.use_compression:
                try:
                    serialized = zlib.decompress(serialized)
                except zlib.error:
                    pass
                    
            data = pickle.loads(serialized)
            
            self.nodes = data["nodes"]
            self.relationships = defaultdict(list)
            for from_node, rels in data.get("relationships", {}).items():
                self.relationships[from_node] = [Relationship(**rel) for rel in rels]
            
            self._indexes = {}
            for prop, index in data.get("_indexes", {}).items():
                self._indexes[prop] = defaultdict(set)
                for value, ids in index.items():
                    self._indexes[prop][value] = set(ids)
                    
            # Инициализация кэшей
            self._select_cache = {}
            self._rel_cache = {}
            
            # Сброс анализа графа
            self._invalidate_graph_analysis()
                    
        except FileNotFoundError:
            print("Файл не найден. Создана новая БД.")
        except Exception as e:
            print(f"Ошибка загрузки: {e}. Создана новая БД.")

    # ================== ТРАНЗАКЦИИ ==================
    
    @contextlib.contextmanager
    def transaction(self):
        """Контекстный менеджер для транзакций"""
        with db_lock:
            backup = {
                "nodes": deepcopy(self.nodes),
                "relationships": deepcopy(dict(self.relationships)),
                "_indexes": deepcopy(self._indexes),
                "_select_cache": deepcopy(self._select_cache),
                "_rel_cache": deepcopy(self._rel_cache),
                "_shortest_paths": deepcopy(self._shortest_paths),
                "_node_ids": deepcopy(self._node_ids),
                "_id_to_idx": deepcopy(self._id_to_idx)
            }
            try:
                yield
                self.save()
            except Exception as e:
                # Откат изменений
                self.nodes = backup["nodes"]
                self.relationships = defaultdict(list, backup["relationships"])
                self._indexes = backup["_indexes"]
                self._select_cache = backup["_select_cache"]
                self._rel_cache = backup["_rel_cache"]
                self._shortest_paths = backup["_shortest_paths"]
                self._node_ids = backup["_node_ids"]
                self._id_to_idx = backup["_id_to_idx"]
                raise e

    # ================== ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ ==================
    
    def print_stats(self):
        """Печать статистики"""
        print("\n=== Статистика базы данных ===")
        print("Операции:")
        for op, count in self.stats['operations'].items():
            print(f"{op}: {count}")
        
        if self.stats['performance']:
            print("\nСреднее время операций (мс):")
            for op, times in self.stats['performance'].items():
                avg = sum(times) / len(times) * 1000
                print(f"{op}: {avg:.2f}")

    def print_nodes(self, limit: int = 10, show_edges: bool = False):
        """Печать информации об узлах"""
        with db_lock:
            print("\n=== Узлы базы данных ===")
            print(f"Общее количество узлов: {len(self.nodes)}")
            
            for i, (node_id, node_data) in enumerate(list(self.nodes.items())[:limit]):
                print(f"\nУзел ID: {node_id}")
                print("Свойства:")
                for prop, value in node_data['props'].items():
                    print(f"  {prop}: {value}")
                
                if show_edges:
                    edges = node_data['edges']
                    if edges:
                        print("Связи:")
                        for target, rel_type in edges:
                            rels = self.find_relationships(from_node=node_id, to_node=target, rel_type=rel_type)
                            for rel in rels:
                                print(f"  -> {target} [{rel_type}]")
                                if rel.props:
                                    print("     Атрибуты связи:", rel.props)
                    else:
                        print("Связи отсутствуют")
            
            if len(self.nodes) > limit:
                print(f"\n... и еще {len(self.nodes) - limit} узлов")

if __name__ == '__main__':
    db = GraphDB("test_db.pickle")
    db.load()  # Загружаем данные (если файл есть)

    # Вставка данных
    db.insert("Alice", {"age": 30, "occupation": "Developer", "city": "Berlin"})
    db.insert("Bob", {"age": 25, "occupation": "Designer", "city": "Paris"})
    db.insert("Charlie", {"age": 35, "occupation": "Manager", "city": "Berlin"})
    db.insert("Diana", {"age": 31, "occupation": "Developer", "city": "London"})

    db.add_relationship("Alice", "Bob", "FRIENDS")
    db.add_relationship("Bob", "Charlie", "WORKS_WITH", {"project": "XYZ"})
    db.add_relationship("Alice", "Diana", "COLLEAGUES")

    # Распечатать всех
    for node in db.nodes:
        print(node )

    # # Примеры запросов
    # print("--- SELECT (age=30) ---")
    # print(db.select(age=30))  # [{'id': 'Alice', 'age': 30, 'occupation': 'Developer'}]

    # print("--- FILTER (age > 28) ---")
    # print(db.filter(lambda props: props.get("age", 0) > 28))

    # print("--- ORDER_BY (age) ---")
    # print(db.order_by("age"))

    # print("--- GROUP_BY (occupation) ---")
    # print(db.group_by("occupation"))

    # # Аналог WHERE age > 25 AND occupation LIKE 'D%'
    # result = [
    #     node for node in db.nodes.values() 
    #     if node["props"]["age"] > 25 
    #     and node["props"]["occupation"].startswith("D")
    # ]
    # print(result)

    # # 1. Найти всех разработчиков из Берлина:
    # devs_in_berlin = db.filter(lambda x: x["occupation"] == "Developer" and x["city"] == "Berlin")
    # print('devs_in_berlin', devs_in_berlin)

    # # 2. Группировка по городу с фильтрацией (HAVING):
    # groups = db.having("city", lambda k, v: len(v) > 1)  # Города с >1 человека
    # print('groups', groups)

    # # 3. Комбинированный SELECT:
    result = db.select(age=30, occupation="Developer")  # Аналог WHERE age=30 AND occupation="Developer"
    print('Комбинированный SELECT', result)

    db.save()  # Сохраняем в файл


