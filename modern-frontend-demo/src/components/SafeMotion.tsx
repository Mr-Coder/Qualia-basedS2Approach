import React from 'react'
import { motion, MotionProps } from 'framer-motion'

// 安全的 motion 组件包装器，防止内部 map 错误
export const SafeMotionDiv = ({ children, ...props }: MotionProps & { children?: React.ReactNode }) => {
  try {
    return <motion.div {...props}>{children}</motion.div>
  } catch (error) {
    console.error('Motion component error:', error)
    return <div>{children}</div>
  }
}

export const SafeMotionPath = ({ ...props }: React.SVGProps<SVGPathElement> & MotionProps) => {
  try {
    return <motion.path {...props} />
  } catch (error) {
    console.error('Motion path error:', error)
    return <path {...props} />
  }
}

export const SafeMotionCircle = ({ ...props }: React.SVGProps<SVGCircleElement> & MotionProps) => {
  try {
    return <motion.circle {...props} />
  } catch (error) {
    console.error('Motion circle error:', error)
    return <circle {...props} />
  }
}

export const SafeMotionText = ({ children, ...props }: React.SVGProps<SVGTextElement> & MotionProps & { children?: React.ReactNode }) => {
  try {
    return <motion.text {...props}>{children}</motion.text>
  } catch (error) {
    console.error('Motion text error:', error)
    return <text {...props}>{children}</text>
  }
}