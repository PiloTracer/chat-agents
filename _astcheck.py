import ast,sys
p='backend/app/llm_provider.py'
with open(p,'r',encoding='utf-8') as f:
    src=f.read()
ast.parse(src)
print('AST OK:', p)
