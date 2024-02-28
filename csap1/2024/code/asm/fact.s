    bits 64

section .note.GNU-stack noalloc noexec nowrite progbits

section .text
    extern printf
    global main

; ----------------------------------------------------------------------
; argument in edi, return in eax
fact:
    mov     eax, 1      ; eax = 1
    cmp     edi, 0      ; if (edi == 0)
    je      fact_end    ;   go to fact_end
fact_loop:
    imul    eax, edi    ; eax = eax * edi
    sub     edi, 1      ; edi = edi - 1
    jnz     fact_loop   ; if (edi != 0) goto fact_loop
fact_end:
    ret                 ; return to caller


; ----------------------------------------------------------------------
; main(), just to call fact() and printf()
main:
    sub     rsp, 8

    mov     edi, 10             ; edi = 10 
    call    fact                ; call fact(10)

    lea     rdi, [rel format]   ; setup the call to printf()
    mov     esi, eax
    xor     eax, eax
    call    printf wrt ..plt

    xor     eax, eax            ; return 0
    add     rsp, 8
    ret

; ----------------------------------------------------------------------
; Read-only data
section .rodata
format: db "fact(10) = %d", 0x0a, 0x00

