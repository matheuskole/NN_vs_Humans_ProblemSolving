import pygame
import sys

#definção das cores
branco = (255, 255, 255)
preto = (0, 0, 0)
azul = (70, 130, 180)
vermelho = (255, 69, 0)

#largura, altura e célula (9 células compõe o jogo da velha)
largura = 600
altura = 600
tam_celula = largura // 3

def desenha_tabuleiro():
    screen.fill(preto)
    for x in range(1, 3):  #linhas verticais
        pygame.draw.line(screen, branco, (x * tam_celula, 0), (x * tam_celula, altura), 3)
    for y in range(1, 3):  #linhas horizontais
        pygame.draw.line(screen, branco, (0, y * tam_celula), (largura, y * tam_celula), 3)
  
#função para desenhar o x numa célula
def desenha_X(x, y):
    padding = 20  #espaçamento interno (ajuste visual)
    start_x = x * tam_celula + padding
    start_y = y * tam_celula + padding
    end_x = (x + 1) * tam_celula - padding
    end_y = (y + 1) * tam_celula - padding
    pygame.draw.line(screen, azul, (start_x, start_y), (end_x, end_y), 8)
    pygame.draw.line(screen, azul, (start_x, end_y), (end_x, start_y), 8)
    
#função que desenha um círculo

def desenha_O(x, y):
    center_x = x * tam_celula + tam_celula // 2
    center_y = y * tam_celula + tam_celula // 2
    raio = tam_celula // 3
    pygame.draw.circle(screen, vermelho, (center_x, center_y), raio, 8)
    
#função para criar a matriz 3x3 do jogo
def jogo_da_velha():
  matriz = [[' ' for _ in range(3)] for _ in range(3)]
  return matriz
  
#checa se o o jogo já terminou
def ganhou(matriz):

    #verificação de linhas e colunas
    for i in range(3):
        if matriz[i][0] == matriz[i][1] == matriz[i][2] != ' ':
            return matriz[i][0] #indica o vencedor

    for j in range(3):
        if matriz[0][j] == matriz[1][j] == matriz[2][j] != ' ':
            return matriz[0][j]

    #verificação das diagonais
    if matriz[0][0] == matriz[1][1] == matriz[2][2] != ' ':
        return matriz[0][0]

    if matriz[0][2] == matriz[1][1] == matriz[2][0] != ' ':
        return matriz[0][2]

    return None #se ainda nao tiver vencedor
    
def mostrar_vencedor(vencedor):
    font = pygame.font.Font(None, 74)
    if vencedor == 'X':
        text = font.render("X venceu!", True, azul)
    elif vencedor == 'O':
        text = font.render("O venceu!", True, vermelho)
    else:
        text = font.render("Deu velha!", True, branco)
    screen.fill(preto)
    screen.blit(text, (largura // 2 - text.get_width() // 2, altura // 2 - text.get_height() // 2))
    pygame.display.update()
    pygame.time.delay(3000)

def jogar():
    #inicializa o pygame e deixa screen como variavel global
    pygame.init()
    global screen
    screen = pygame.display.set_mode((largura, altura))
    pygame.display.set_caption("Jogo da Velha")
    
    matriz = jogo_da_velha()  #inicializo a matriz
    desenha_tabuleiro()
    running = True  
    jogador_atual = 'X'
    
    #loop controle
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                linha = y // tam_celula
                coluna = x // tam_celula

                if matriz[linha][coluna] == ' ':
                    matriz[linha][coluna] = jogador_atual

                    if jogador_atual == 'X':
                        desenha_X(coluna, linha)
                        jogador_atual = 'O'
                    else:
                        desenha_O(coluna, linha)
                        jogador_atual = 'X'

                    #faz checagem se alguém já ganhou
                    vencedor = ganhou(matriz)
                    if vencedor:
                        mostrar_vencedor(vencedor)
                        matriz = jogo_da_velha()  #reincializa
                        desenha_tabuleiro()  #desenha novamente para outra rodada
                        jogador_atual = 'X'

                    #verifica se deu velha(empate)
                    empate = True
                    for linha in matriz:
                        for celula in linha:
                            if celula == ' ':
                                empate = False  #se tiver alguma vazia, nao é empate
                                break
                        if not empate:
                            break
                    
                    if empate:
                        mostrar_vencedor(None)  # Mostra que deu empate (velha)
                        matriz = jogo_da_velha()  # Reinicia a matriz
                        desenha_tabuleiro()  # Redesenha o tabuleiro
                        jogador_atual = 'X'  # X começa jogando

        pygame.display.update()
    
    
if __name__ == "__main__":
    jogar() 