# Инструкции по настройке: GitHub + Obsidian

---

## ЧАСТЬ 1: Создание GitHub репозитория

### Шаг 1: Создать аккаунт и репозиторий на GitHub

1. Зайди на [github.com](https://github.com) (аккаунт уже должен быть)
2. Нажми **"+"** в правом верхнем углу → **"New repository"**
3. Заполни:
   - Repository name: `enterprise-ai-engineer`
   - Description: `9-Month Enterprise AI Agent Engineer Training Program`
   - Visibility: **Private** (это личный учебный материал)
   - НЕ ставь галочки на Initialize, .gitignore, license — мы добавим всё сами
4. Нажми **"Create repository"**

### Шаг 2: Настроить Git на сервере лаборатории

Подключись к серверу по SSH и выполни:

```bash
# Установить git (скорее всего уже есть)
sudo apt-get install git -y

# Настроить имя и email (используй те же что на GitHub)
git config --global user.name "Vladimir"
git config --global user.email "твой@email.com"

# Настроить ветку по умолчанию
git config --global init.defaultBranch main
```

### Шаг 3: Настроить SSH-ключ для GitHub

Это позволит пушить без пароля каждый раз:

```bash
# Сгенерировать SSH ключ
ssh-keygen -t ed25519 -C "твой@email.com"
# Нажимай Enter на все вопросы (путь по умолчанию, без passphrase)

# Показать публичный ключ — его нужно скопировать
cat ~/.ssh/id_ed25519.pub
```

Скопируй весь вывод (начинается с `ssh-ed25519`).

На GitHub: Settings → SSH and GPG keys → New SSH key → вставь ключ → Save.

Проверь что всё работает:
```bash
ssh -T git@github.com
# Должно вывести: "Hi username! You've successfully authenticated..."
```

### Шаг 4: Создать локальный репозиторий и запушить

```bash
# Создать папку проекта
mkdir ~/enterprise-ai-engineer
cd ~/enterprise-ai-engineer

# Инициализировать git
git init

# Скопируй все файлы которые я создал в эту папку
# (они придут как скачиваемый архив или по отдельности)

# Добавить все файлы в git
git add .

# Первый коммит
git commit -m "Initial structure: M0-M7 repo skeleton + lab passport + ADR-001"

# Подключить к GitHub (замени USERNAME на твой логин)
git remote add origin git@github.com:USERNAME/enterprise-ai-engineer.git

# Запушить
git push -u origin main
```

---

## ЧАСТЬ 2: Настройка Obsidian

Obsidian работает с локальной папкой — твой репозиторий и есть его "vault" (хранилище).

### Шаг 1: Открыть папку репозитория как Vault

1. Открой Obsidian
2. На стартовом экране: **"Open folder as vault"**
3. Выбери папку `enterprise-ai-engineer` (ту же что создал для GitHub)
4. Obsidian спросит про Trust — нажми **"Trust and Enable All"**

Теперь Obsidian видит все `.md` файлы репозитория как заметки.

### Шаг 2: Базовые настройки Obsidian

Settings (Ctrl+,):

**Editor:**
- Default view mode → **Reading** (читать по умолчанию, редактировать по Ctrl+E)
- Show line numbers → ON
- Spell check → OFF (или настрой русский словарь)

**Files & Links:**
- Default location for new notes → **Same folder as current file**
- New link format → **Relative path to file**

**Appearance:**
- Theme → поставь **Minimal** или **Things** (красивее стандартного)

### Шаг 3: Полезные горячие клавиши

| Действие | Клавиши |
|---------|--------|
| Переключить режим (чтение/редактирование) | Ctrl + E |
| Открыть файл по имени | Ctrl + O |
| Поиск по всем заметкам | Ctrl + Shift + F |
| Создать новую заметку | Ctrl + N |
| Раскрыть/свернуть боковую панель | Ctrl + B |

### Шаг 4: Плагины (опционально, но полезно)

Settings → Community plugins → Browse:

- **Git** — авто-коммит и пуш прямо из Obsidian (очень удобно!)
- **Dataview** — создавать таблицы из метаданных заметок (например, статус всех заданий)
- **Templater** — шаблоны для новых заметок

**Настройка плагина Git:**
После установки: Settings → Git → Vault backup interval → 30 минут.
Он будет автоматически делать `git commit` и `git push` каждые 30 минут.

---

## ЧАСТЬ 3: Рабочий процесс

### Как работать с конспектом в процессе курса

```
Учишь → Пишешь в Obsidian → Git plugin пушит на GitHub автоматически
```

Конкретно для каждого задания:
1. Открываешь `M0-Foundations/notes.md` в Obsidian
2. Записываешь что делал и что понял своими словами
3. Код кладёшь в `M0-Foundations/src/`
4. Git plugin сам пушит (или делаешь вручную: `git add . && git commit -m "M0: task 0.1 done" && git push`)

### Соглашение по коммит-сообщениям (commit messages)

```
M0: task 0.1 - first llm call working
M0: notes - added tool use explanation
M1: src - llm gateway v0.1
ADR-002: added gateway architecture decision
```

Формат: `[Модуль]: [что сделано]` — через 3 месяца будет понятно что было в каждом коммите.

---

## Что делать дальше

После того как GitHub repo создан и Obsidian настроен:

1. Убедись что файлы из скачанного архива лежат в правильных папках
2. Сделай первый `git push` и проверь что всё появилось на GitHub
3. Открой `M0-Foundations/notes.md` в Obsidian и убедись что рендерится нормально
4. Сообщи мне — и начинаем первое задание M0
