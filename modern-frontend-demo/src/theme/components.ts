import { defineStyleConfig } from '@chakra-ui/react'

const Button = defineStyleConfig({
  baseStyle: {
    fontWeight: 'semibold',
    borderRadius: 'xl',
    transition: 'all 0.2s',
  },
  sizes: {
    sm: {
      fontSize: 'sm',
      px: 4,
      py: 2,
      h: 8,
    },
    md: {
      fontSize: 'md',
      px: 6,
      py: 3,
      h: 10,
    },
    lg: {
      fontSize: 'lg',
      px: 8,
      py: 4,
      h: 12,
    },
  },
  variants: {
    math: {
      bg: 'linear-gradient(135deg, #a855f7 0%, #3b82f6 100%)',
      color: 'white',
      _hover: {
        transform: 'scale(1.05)',
        boxShadow: 'math',
        _disabled: {
          transform: 'none',
        },
      },
      _active: {
        transform: 'scale(0.95)',
      },
    },
    reasoning: {
      bg: 'reasoning.500',
      color: 'white',
      _hover: {
        bg: 'reasoning.600',
        boxShadow: 'reasoning',
      },
    },
    entity: {
      bg: 'semantic.entity',
      color: 'white',
      _hover: {
        bg: 'cyan.600',
        transform: 'translateY(-2px)',
      },
    },
    relation: {
      bg: 'semantic.relation',
      color: 'white',
      _hover: {
        bg: 'lime.600',
        transform: 'translateY(-2px)',
      },
    },
  },
  defaultProps: {
    size: 'md',
    variant: 'solid',
  },
})

const Card = defineStyleConfig({
  baseStyle: {
    p: 6,
    borderRadius: 'xl',
    bg: 'white',
    boxShadow: 'md',
    border: '1px solid',
    borderColor: 'gray.200',
  },
  variants: {
    math: {
      borderColor: 'math.200',
      borderWidth: '2px',
      bg: 'white',
      _hover: {
        boxShadow: 'math',
        borderColor: 'math.300',
      },
    },
    reasoning: {
      borderColor: 'reasoning.200',
      borderWidth: '2px',
      bg: 'white',
      _hover: {
        boxShadow: 'reasoning',
        borderColor: 'reasoning.300',
      },
    },
    elevated: {
      boxShadow: 'xl',
      bg: 'white',
    },
  },
})

const Input = defineStyleConfig({
  variants: {
    math: {
      field: {
        borderColor: 'math.300',
        borderWidth: '2px',
        borderRadius: 'lg',
        _focus: {
          borderColor: 'math.500',
          boxShadow: '0 0 0 1px var(--chakra-colors-math-500)',
        },
        _hover: {
          borderColor: 'math.400',
        },
      },
    },
  },
})

const Textarea = defineStyleConfig({
  variants: {
    math: {
      borderColor: 'math.300',
      borderWidth: '2px',
      borderRadius: 'lg',
      _focus: {
        borderColor: 'math.500',
        boxShadow: '0 0 0 1px var(--chakra-colors-math-500)',
      },
      _hover: {
        borderColor: 'math.400',
      },
    },
  },
})

const Badge = defineStyleConfig({
  variants: {
    complexity: {
      borderRadius: 'full',
      px: 3,
      py: 1,
      fontSize: 'xs',
      fontWeight: 'bold',
    },
    entity: {
      bg: 'semantic.entity',
      color: 'white',
      borderRadius: 'md',
    },
    relation: {
      bg: 'semantic.relation',
      color: 'white',
      borderRadius: 'md',
    },
    operation: {
      bg: 'semantic.operation',
      color: 'white',
      borderRadius: 'md',
    },
  },
})

export const components = {
  Button,
  Card,
  Input,
  Textarea,
  Badge,
}