## Контекстные менеджеры

`__enter__` `__exit__` `context manager` `with`

### Условие

Контекстные менеджеры позволяют управлять ресурсами и изменять поведение кода внутри блока `with`. В этой задаче вы реализуете несколько полезных контекстных менеджеров с использованием методов `__enter__` и `__exit__`.

#### Timer

```python
with Timer() as timer:
    some_operation()
print(f"Elapsed: {timer.elapsed} seconds")
```

Замеряет время выполнения кода внутри блока `with`. После завершения блока в атрибуте `timer.elapsed` доступно прошедшее время в секундах.

#### FileManager

```python
with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')
# Файл автоматически закрыт при выходе из блока
```

Управляет файлом: открывает его при входе в блок и гарантирует закрытие при выходе (даже если произошло исключение). Это упрощённая версия встроенной функции `open()` для обучения механике контекстных менеджеров.

#### OutputCapture

```python
with OutputCapture() as captured:
    print("Hello")
    print("World", file=sys.stderr)
print(captured.stdout)  # "Hello\n"
print(captured.stderr)  # "World\n"
```

Перехватывает вывод в stdout и stderr внутри блока `with`. После завершения блока перехваченный вывод доступен в атрибутах `captured.stdout` и `captured.stderr`.

### Про задачу

Контекстный менеджер — это объект с методами `__enter__` и `__exit__`:

```python
class MyContextManager:
    def __enter__(self):
        # Выполняется при входе в блок with
        # Обычно здесь происходит сохранение состояния или setup
        return self  # или другой объект для использования в "with ... as"

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Выполняется при выходе из блока with
        # Обычно здесь происходит восстановление состояния или cleanup
        # Если вернуть True, исключение будет подавлено
        return False
```

**Паттерн реализации:**

1. **`__init__`**: сохраняем параметры для дальнейшего использования
2. **`__enter__`**: сохраняем текущее состояние, выполняем setup, возвращаем `self` или нужный объект
3. **`__exit__`**: восстанавливаем состояние, выполняем cleanup, возвращаем `False` (чтобы исключения пробрасывались дальше)

В этой задаче вы практикуете базовую механику контекстных менеджеров без сложной обработки исключений.

### Проверка

После реализации контекстных менеджеров попробуйте запустить следующие примеры:

**Timer:**
```python
# Замер времени выполнения функции
def slow_function():
    import time
    time.sleep(0.1)
    return 42

with Timer() as timer:
    result = slow_function()
print(f"Result: {result}, Time: {timer.elapsed:.2f}s")
```

**FileManager:**
```python
# Запись и чтение файла
import os
filename = 'test_demo.txt'

with FileManager(filename, 'w') as f:
    f.write('Line 1\n')
    f.write('Line 2\n')

with FileManager(filename, 'r') as f:
    for line in f:
        print(f"Read: {line.strip()}")

os.remove(filename)  # cleanup
```

**OutputCapture:**
```python
# Перехват вывода функции
def print_report():
    print("Report header")
    print("Error: something happened", file=sys.stderr)
    print("Report footer")

with OutputCapture() as captured:
    print_report()

print("Captured stdout:", repr(captured.stdout))
print("Captured stderr:", repr(captured.stderr))
```

**Вложенные контекстные менеджеры:**
```python
# Можно вкладывать контекстные менеджеры друг в друга
with Timer() as outer_timer:
    with OutputCapture() as captured:
        print("Inside both managers")
        time.sleep(0.05)

print(f"Time: {outer_timer.elapsed:.2f}s")
print(f"Captured: {captured.stdout.strip()}")
```

### Уточнения

* Используйте `time.time()` или `time.perf_counter()` для замера времени в Timer
* Для FileManager используйте встроенную функцию `open()` и метод `.close()`
* Для OutputCapture используйте `io.StringIO` для перехвата вывода и `sys.stdout`/`sys.stderr` для доступа к потокам
* Все контекстные менеджеры должны быть реализованы как классы с методами `__enter__` и `__exit__`
* `__exit__` должен корректно работать даже при возникновении исключений внутри блока `with`
