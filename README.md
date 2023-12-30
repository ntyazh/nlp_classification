В ходе работы были предложены два принципиально разных подхода к решению задачи классификации комментариев о работе сети магазинов по их темам.

С помощью классического ML. В этой части была проведена токенизация, полученные токены были лемматизированы и преобразованы в
векторы с помощью обученной модели Word2Vec. Далее была решена проблема несбалансированности классов с помощью SMOTE, 
который создаёт новые объекты данных в непосредственной близости от уже существующих в наименьшем классе. 
После этого был построен ансамбль, состоящий из алгоритма RandomForest, линейного классификатора и двух алгоритмов бустинга: 
lightGBM и XGB. Для каждой модели были подобраны гиперпараметры с помощью GridSearchCV. f1-мера на тестовой выборке составила 0.84.

С помощью DL. Был произведён fine-tuning предобученного на текстах на русском языке также на задачу классификации трансформера BERT. 
В рамках этой части были написаны кастомные датасет, класс для классификации BertClassifier, функции обучения и тестирования нейронной сети. 
Был подобран learning_rate для используемого оптимизатора AdamW и также был выбран шедулер. f1-мера на тестовой выборке составила 0.93. 
Лучшая модель была сохранена в файл bert.pth, а также был написан её инференс — функция get_result в файле model.py.
