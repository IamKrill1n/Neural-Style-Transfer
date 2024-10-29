# Neural-Style-Transfer
---
## Github Commit Flow:
#### 1, Clone Repo:
```bash
git clone https://github.com/IamKrill1n/Neural-Style-Transfer.git
```

#### 2, (Local) Pull $${\color{lightblue}Main \space Branch}$$ to your $${\color{lightblue}Local \space Main \space Branch}$$ before do anything!:
```bash
git checkout main
git pull origin main
```

#### 3, (Local) Create/Switch to your $${\color{lightblue}Local}$$ branch named ***features/{feature_name}***:
- Create (Skip this if you have this branch):
```bash
git checkout -b features/{feature_name}
```

- Switch:
```bash
git checkout features/{feature_name}
```

#### 4, (Local) Add, Commit your changes:
- Add:
($${\color{lightblue}Note}$$: '.' represent all files)
```bash
git add .
```

- Commit:
($${\color{lightblue}Note}$$: The ***{message}*** is recommended to be in the syntax "{Action} {Object})
```bash
git commit -m {message}
```

#### 5, (Local) Push your changes:
- Create Remote Branch:
($${\color{lightblue}Note}$$: Skip this if this REPOSITORY have your ***features/{feature_name}*** branch)
```bash
git push origin --set-upstream features/{feature_name}
```

- Push:
```bash
git push origin features/{feature_name}
```

#### 6, (Remote) Check the Pull Request and Fix the errors (if exists)
---
## To-do List

### Experiment with Hyperparameters in the Original Method and other Improvemnents
- Change the layers and layers' weights and observe the effects.
- Adjust the content/style weights (alpha/beta), may be try Patch-based Style loss.
- Modify the initialization, for example:
    - White noise
    - Content image
    - Blurred content image
    - Partially stylized image

### Explore Other Methods
- Read the survey and try other methods: [Neural Style Transfer: A Review](https://arxiv.org/pdf/1705.04058)
- Read the survey about evaluation: [Evaluation in Neural Style Transfer: A Review](https://arxiv.org/pdf/2401.17109)